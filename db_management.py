"""
Utility functions for working with databases.

Main databases:
- files.db : stores file hashes to check if file uploaded already exists.
"""

import streamlit as st
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
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import List, Tuple, Dict, Set, Iterable

from constants import *
from utils import *

def list_collection_names(chroma_client : chromadb.PersistentClient) -> List[str]:
    """
    List names of all ChromaDB collections.
    """
    collection_names = []
    for collection in chroma_client.list_collections():
        collection_names.append(collection.name)
    return collection_names

def setup_database() -> None:
    """
    Produce tables if not already existing.
    """
    conn = sqlite3.connect(FILES_DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS collections (
        collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT UNIQUE NOT NULL
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        UNIQUE(file_hash)
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS collection_files (
        collection_id INTEGER,
        file_id INTEGER,
        UNIQUE(collection_id, file_id),
        FOREIGN KEY (collection_id) REFERENCES collections (collection_id),
        FOREIGN KEY (file_id) REFERENCES files (file_id)
    )''') 

    conn.commit()
    conn.close()

def files_upload_pipeline(
    collection_name : str,
    openai_client : openai.OpenAI,
    chroma_client : chromadb.PersistentClient,
    embedding_function : OpenAIEmbeddingFunction,
    pdf_files : List[BytesIO]
) -> None:
    """
    Given a list of PDF files, add each file to the database, if it does not
    already exist in the specified collection `collection_name`.
    """
    chunked_documents = []
    conn = sqlite3.connect(FILES_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT collection_id FROM collections WHERE collection_name = ?", (collection_name,))
    collection_id = cursor.fetchone()[0]

    for pdf_file in pdf_files:

        file_hash = hashlib.sha256(pdf_file.read()).hexdigest()

        # Returns all files with the same file hash in the current collection
        cursor.execute(''' 
        SELECT cf.file_id FROM collection_files cf
        JOIN files f ON f.file_id = cf.file_id
        WHERE cf.collection_id = ? AND f.file_hash = ?''', (collection_id, file_hash))

        # If the file is already in the collection, skip the rest of the loop
        if cursor.fetchone():
            continue
        
        # Insert the file hash into files table
        cursor.execute("INSERT OR IGNORE INTO files (file_hash) VALUES (?)", (file_hash,))
        conn.commit()

        # Insert collection ID and files ID into collection_files table
        cursor.execute("SELECT file_id FROM files WHERE file_hash = ?", (file_hash,))
        file_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (collection_id, file_id))
        conn.commit()

        # Extract questions
        chunked_document = extract_text_from_pdf(openai_client, pdf_file)
        chunked_documents += chunked_document

        st.sidebar.success(f"Uploaded {pdf_file.name} to {collection_name}.")

    # Embed these chunks with ChromaDB
    if len(chunked_documents) > 0:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function)
        collection.add(
            documents=chunked_documents,
            ids=[str(uuid.uuid4()) for _ in range(len(chunked_documents))],
        )
        st.sidebar.success(f"Added all documents to {collection_name}")
    
    conn.close()
