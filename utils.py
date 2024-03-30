"""
Utility functions.
"""

import openai 
import os 
import re 
import pandas as pd 
import PyPDF2
import pprint
import numpy as np
import sqlite3
import hashlib
import pickle
from io import BytesIO
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Set, Iterable

from constants import *
from db_management import *

def perform_embedding(client : openai.OpenAI, chunks : List[str]) -> None:
    """
    Given a list of text chunks (`chunked_documents`),
    embed the text chunks and store on the vector database.
    """

    raw_chunk_embeddings = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )

    chunk_embeddings = [e.embedding for e in raw_chunk_embeddings.data]

    add_embeddings_to_vdb(
        new_chunk_embeddings=chunk_embeddings,
        new_chunks=chunks,
        index_file=VECTOR_DB_PATH,
        mapping_file=EMBEDDINGS_TO_TEXT_PATH,
    )

def extract_text_from_pdf(client : openai.OpenAI, pdf_data : BytesIO) -> str:
    """
    Given the binary data of a pdf file, return a list of all the questions in it.
    """

    # Open PDF in read-binary format and extract data iteratively per page
    reader = PyPDF2.PdfReader(pdf_data)

    questions = []

    for page_num in range(len(reader.pages)):

        page_text = reader.pages[page_num].extract_text()

        clean_input_instructions = f"""
        {page_text}

        These are exam questions from a document. 
        Your task is to cleanly separate them into questions.
        Indicate the start of each question with QUESTION START. Do not remove the question description, and do not consider question subparts as separate questions.
        If there are no exam questions in the provided text, return NULL.

        Here's an example of what you should do:

        Original exam question: 
        1) A ball is thrown up into the air with speed v. Neglect air resistance for this question.

        a) What is the maximum height reached by the ball  PAGE3OF5%£"%"  ?
        b) If there is a drag force of kv resisting the  ball, what is the maximum height now? VERSION1OF2PAGE1

        Cleaned question:
        QUESTION START
        1) A ball is thrown up into the air with speed v. Neglect air resistance for this question.

        a) What is the maximum height reached by the ball?
        b) If there is a drag force of kv resisting the ball, what is the maximum height now?
        """

        cleaned_page_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role" : "system", "content" : CLEAN_INPUT_SYSPROMPT},
                {"role" : "user", "content" : clean_input_instructions}
            ],
            temperature=0.8,
        )
        cleaned_page = cleaned_page_response.choices[0].message.content

        if "NULL" not in cleaned_page:
            parts = cleaned_page.split("QUESTION START")
            page_questions = [part.strip() for part in parts if part.strip()]
            questions += page_questions

    return questions

def files_upload_pipeline(
    db_connection : sqlite3.Connection,
    cursor : sqlite3.Cursor,
    client : openai.OpenAI,
    pdf_files : List[BytesIO]
) -> None:
    """
    Given a list of binary data of PDFs, return a list of strings,
    where each string represents a chunk of text from the PDFs.

    Only do this if the documents are not on the database.
    """

    chunked_documents = []

    for pdf_file in pdf_files:
        file_hash = compute_hash(pdf_file)
        if not file_exists(cursor, file_hash): # Check if file hash is in database
            insert_file(cursor, file_hash)
            db_connection.commit()
            print("File was successfully added to database.")
            chunked_document = extract_text_from_pdf(client, pdf_file)
            chunked_documents += chunked_document
        else:
            # Embeddings already exist, so we're good here
            print("File already exists in database!")

    # Embed these chunks
    if len(chunked_documents) > 0:
        perform_embedding(
            client=client,
            chunks=chunked_documents
        )

def convert_latex_format(math_string : str) -> str:
    """
    Given a string written in the \(, \[ LaTeX format,
    return it as a $, $$ formatted LaTeX string.
    """
    # Replace inline math from \(...\) to $...$
    inline_converted = re.sub(r'\\\((.*?)\\\)', r'$\1$', math_string)
    
    # Replace display math from \[...\] to $$...$$
    display_converted = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', inline_converted)
    
    return display_converted

def keywords_pipeline(client : openai.OpenAI, query : str) -> str:
    """
    Returns relevant keywords given a query.

    Inputs:
    - client : the OpenAI client instance being used.
    - query : the user's raw query, without any augmentation.
    """

    keywords_input = f"""
    I am going to provide you with a sentence. Based on it, you must return 10 words similar to the key words in this sentence.

    For example, for the sentence, "Generate an exam problem on thermodynamics", the keyword is "thermodynamics", so you might return "enthalpy, entropy, carnot cycle, heat engine, adiabatic, reverislble, irreversible, clausius, pressure, volume".

    Here is the sentence: {query}
    """

    keywords_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : KEYWORDS_SYSPROMPT},
            {"role" : "user", "content" : keywords_input}
        ],
        temperature=TEMPS["keywords"],
    )
    keywords = keywords_response.choices[0].message.content
    
    return keywords

def question_solution_pipeline(client : openai.OpenAI, query_with_keywords_examples : str) -> Tuple[str, str]:
    """
    Produce a problem and solution on a given academic topic.

    Inputs:
    - client : the OpenAI client instance being used.
    - augmented_query : a str containing the user query and relevant context for the model.
    """

    # Produced question
    question_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : QUESTION_SYSPROMPT},
            {"role" : "user", "content" : query_with_keywords_examples}
        ],
        temperature=TEMPS["question"],
    )
    question = question_response.choices[0].message.content

    answer_instructions = f"""
    Explain in detail your solution to the following exam problem, as if you were an extremely intelligent student taking the exam.

    Problem:
    {question}
    """

    answer_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : ANSWER_SYSPROMPT},
            {"role" : "user", "content" : answer_instructions}
        ],
        temperature=TEMPS["answer"],
    )
    answer = answer_response.choices[0].message.content
    
    question = convert_latex_format(question)
    answer = convert_latex_format(answer)

    return question, answer

def augment_query(client : openai.OpenAI, query : str) -> str:
    """
    Given a user query, augment the query with keywords and example questions.
    """

    keywords = keywords_pipeline(client, query)
    query_with_keywords = f"""
User query: {query}

Relevant keywords: {keywords}
    """

    # Embed user query
    raw_query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_with_keywords,
    )
    query_embedding = [e.embedding for e in raw_query_embedding.data]

    context = query_similar_chunks(
        query_embedding=query_embedding[0],
        n_results=N_CONTEXT,
    )

    formatted_context = ""
    for i, problem in enumerate(context, start=1):
        formatted_context += f"Example problem {i}: {problem}\n\n"
    formatted_context = formatted_context.strip()

    augmented_query = f"""
User query: {query}

Relevant keywords: {keywords}

Draw inspiration from the following example questions:

{formatted_context}
    """

    return augmented_query


    
