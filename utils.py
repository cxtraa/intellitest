"""
Utility functions for use in main application.
"""

import streamlit as st
import torch as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import PyPDF2
import pprint

from openai import OpenAI
from IPython.display import display, Markdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(uploaded_file):
    """
    Given a Streamlit uploaded file, return it as raw text.
    """
    file_stream = io.BytesIO(uploaded_file.getvalue())
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

def clean_and_chunk_text(text):
    """
    Given a string of text, clean it by doing the following:
    - Remove whitespace
    - Remove some special characters
    Return the text as a list of paragraphs (list of strings).
    """
    cleaned_text = re.sub(r'\s+', ' ', text)
    pattern = re.compile(r'(\(\w\))')
    chunks = pattern.split(cleaned_text)
    merged_chunks = []
    for i in range(1, len(chunks), 2):
        merged_chunks.append(f'{chunks[i]} {chunks[i+1].strip()}')
    return merged_chunks

def get_embeddings(chunks, model):
    """
    Given a list of strings (chunks), return the chunk embeddings.
    """
    return model.encode(chunks)

def retrieve_context(query_embedding, chunk_embeddings, chunks, N):
    """
    Given a query embedding, and chunk embeddings, return the N most relevant chunks to the query.
    """
    similarities = cosine_similarity([query_embedding], chunk_embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]
    relevant_chunks = [chunks[i] for i in sorted_indices[:N]]
    return relevant_chunks

def convert_latex_format(math_string):
    """
    Given a string written in the \(, \[ LaTeX format,
    return it as a $, $$ formatted LaTeX string.
    """
    # Replace inline math from \(...\) to $...$
    inline_converted = re.sub(r'\\\((.*?)\\\)', r'$\1$', math_string)
    
    # Replace display math from \[...\] to $$...$$
    display_converted = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', inline_converted)
    
    return display_converted

