"""
Frontend of application.
"""

# Import libraries
import streamlit as st
import os
import re
import numpy as np
import openai
import PyPDF2
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Tuple

from utils import *
from constants import *

# Check session state
if "init_done" not in st.session_state:
    st.session_state["init_done"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_query" not in st.session_state:
    st.session_state["first_query"] = True # checks if this is the user's first query

# Load style.css
with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# Display title
st.title("Intellitest.")

# Recall message history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# API key input
if "openai_api_key" not in st.session_state or st.session_state["openai_api_key"] is None:
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state["openai_api_key"] = api_key
        st.rerun()
    else:
        st.stop()

# Initialise OpenAI client
openai_client = openai.OpenAI(api_key=st.session_state["openai_api_key"])
embedding_function = OpenAIEmbeddingFunction(
    api_key=st.session_state["openai_api_key"],
    model_name=EMBEDDING_MODEL,
)

# Initialise ChromaDB client (creates if not already existing)
chroma_client = chromadb.PersistentClient(path="./chroma")
documents_collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function,
)

# Upload documents
if not st.session_state["init_done"]:

    with st.chat_message("assistant", avatar="🌐"):
        st.markdown(WELCOME)

    uploaded_files = st.file_uploader("Upload your exam documents.", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:  
        
        # Start a connection to the database files.db
        files_db_connection = sqlite3.connect(FILES_DB_PATH)
        cursor = files_db_connection.cursor()

        # Chunk documents
        files_upload_pipeline(
            db_connection=files_db_connection,
            cursor=cursor,
            client=openai_client,
            collection=documents_collection,
            pdf_files=uploaded_files,
        )

        # Commit changes and close
        files_db_connection.commit()
        files_db_connection.close()

        st.session_state["init_done"] = True
        st.success("Documents processed successfully.")
        st.rerun()

# Query model
if st.session_state["init_done"]:

    # Welcome message for user
    if st.session_state["first_query"]:
        with st.chat_message("assistant", avatar="🌐"):
            st.markdown(START_QUERYING) # help message for user if it's their first query
            st.session_state["first_query"] = False

    # Prompt user for input
    query = st.chat_input("Type your query...")

    if query:

        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        st.session_state.messages.append({
            "role" : "user",
            "content" : query,
            "avatar" : "👤",
        })

        # Augment the user's query
        augmented_query = augment_query(
            client=openai_client,
            collection=documents_collection,
            query=query,
        )
        
        # For debugging
        print(augmented_query)

        # Generate question and solution
        question, solution = question_solution_pipeline(
            client=openai_client,
            query_with_keywords_examples=augmented_query,
        )

        with st.chat_message("assistant", avatar="🌐"):
            st.markdown(question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": question,
            "avatar" : "🌐",
            })
        
        with st.chat_message("assistant", avatar="🌐"):
            st.markdown(solution)
        st.session_state.messages.append({
            "role": "assistant",
            "content": solution,
            "avatar" : "🌐",
            })