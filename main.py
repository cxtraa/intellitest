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
from db_management import *
from ui import *

# Check session state
if "init_done" not in st.session_state:
    st.session_state["init_done"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_query" not in st.session_state:
    st.session_state["first_query"] = True # checks if this is the user's first query
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = None

# Load style.css
with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# Setup database
setup_database()

# Display title
st.title("Intellitest.")

# Recall message history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# API key input
if "openai_api_key" not in st.session_state or st.session_state["openai_api_key"] is None:
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
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

# Initialise ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma")

with st.chat_message("assistant", avatar="🌐"):
    st.markdown(WELCOME)

if len(list_collection_names(chroma_client)) == 0:
    st.sidebar.subheader("No collections. Please create one.")
    create_collection_ui(
        chroma_client=chroma_client,
        embedding_function=embedding_function,
    )
else:
    collection_and_upload_ui(
        openai_client=openai_client,
        chroma_client=chroma_client,
        embedding_function=embedding_function,
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Create a new collection.")
    create_collection_ui(
        chroma_client=chroma_client,
        embedding_function=embedding_function,
    )

# Prompt user for input
query = st.chat_input("Type your query...")

if query and st.session_state["collection_name"]:

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    st.session_state.messages.append({
        "role" : "user",
        "content" : query,
        "avatar" : "👤",
    })

    # Augment the user's query
    augmented_query = augment_query(
        openai_client=openai_client,
        chroma_client=chroma_client,
        collection_name=st.session_state["collection_name"],
        query=query,
    )
    
    # For debugging
    print(augmented_query)

    # Generate question and solution
    question, solution = question_solution_pipeline(
        openai_client=openai_client,
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