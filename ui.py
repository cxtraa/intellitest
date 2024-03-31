"""
Functions for managing the UI.
"""

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

def create_collection_ui(
        chroma_client : chromadb.PersistentClient,
        embedding_function : OpenAIEmbeddingFunction) -> None:
    """
    UI procedure for creating a new collection.
    """
    new_collection_name = st.sidebar.text_input("Enter the collection name:")
    create_new_collection = st.sidebar.button("Create a new collection")

    if create_new_collection:

        if new_collection_name in list_collection_names(chroma_client):
            st.sidebar.error("This collection already exists.")

        else:
            
            # Add the collection to the database
            conn = sqlite3.connect(FILES_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO collections (collection_name) VALUES (?)", (new_collection_name,))
            conn.commit()
            conn.close()

            # Create the new ChromaDB collection
            chroma_client.create_collection(
                name=new_collection_name,
                embedding_function=embedding_function,
            )

            st.sidebar.success(f"Collection {new_collection_name} was created!")
            st.rerun()

def collection_and_upload_ui(
        openai_client : openai.OpenAI,
        chroma_client : chromadb.PersistentClient,
        embedding_function : OpenAIEmbeddingFunction) -> None:
    """
    UI procedure for selecting a collection and uploading documents.
    """
    collection_name = st.sidebar.selectbox("Select a Collection", options=list_collection_names(chroma_client))

    if collection_name:
        st.session_state["collection_name"] = collection_name
        uploaded_files = st.sidebar.file_uploader("Upload documents", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:

            # Chunk documents
            files_upload_pipeline(
                collection_name=collection_name,
                openai_client=openai_client,
                chroma_client=chroma_client,
                embedding_function=embedding_function,
                pdf_files=uploaded_files,
            )

    