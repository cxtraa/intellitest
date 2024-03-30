"""
Utility functions for working with databases.

Main databases:
- files.db : stores file hashes to check if file uploaded already exists.
- chunks.ann : stores vector embeddings for context retrieval.
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

def query_similar_chunks(query_embedding, n_results=5, index_file=VECTOR_DB_PATH, mapping_file=EMBEDDINGS_TO_TEXT_PATH):
    """
    Given a query embedding, return the top `n_results` text chunks
    from the vector database, using the embeddings to text dictionary file.
    """
    # Load the index
    t = AnnoyIndex(len(query_embedding), 'angular')
    t.load(index_file)  # Super fast, will just mmap the file
    
    # Load the mapping
    with open(mapping_file, 'rb') as f:
        chunks = pickle.load(f)
    
    # Find the n most similar items
    nearest_ids = t.get_nns_by_vector(query_embedding, n_results, include_distances=False)
    
    # Retrieve the corresponding text chunks
    similar_chunks = [chunks[i] for i in nearest_ids]
    return similar_chunks

def add_embeddings_to_vdb(new_chunk_embeddings, new_chunks, index_file='chunks.ann', mapping_file='chunk_mapping.pkl'):
    """
    Adds embeddings to the vector database.
    """

    # Check if the index and mapping files already exist
    index_exists = os.path.isfile(index_file)
    mapping_exists = os.path.isfile(mapping_file)
    
    if index_exists and mapping_exists:
        # Load the existing index and mapping
        with open(mapping_file, 'rb') as f:
            chunks = pickle.load(f)
        embedding_dim = len(new_chunk_embeddings[0])
        t = AnnoyIndex(embedding_dim, 'angular')
        t.load(index_file)
        initial_index = len(chunks)  # Starting index for new embeddings
    else:
        # Create a new index and mapping
        chunks = []
        embedding_dim = len(new_chunk_embeddings[0])
        t = AnnoyIndex(embedding_dim, 'angular')
        initial_index = 0  # Starting from scratch
    
    # Add new embeddings and chunks to the index and mapping
    for i, embedding in enumerate(new_chunk_embeddings, start=initial_index):
        t.add_item(i, embedding)
        chunks.append(new_chunks[i - initial_index])
    
    t.build(10)  # Rebuild the index
    t.save(index_file)  # Save the updated index
    
    # Save the updated mapping
    with open(mapping_file, 'wb') as f:
        pickle.dump(chunks, f)

def compute_hash(pdf_file : BytesIO):
    """
    Given the path to a file, compute its hash.
    """

    # Set the pointer to the beginning of the file
    pdf_file.seek(0)
    file_hash = hashlib.sha256(pdf_file.read()).hexdigest()
    pdf_file.seek(0)

    return file_hash

def file_exists(cursor, file_hash):
    """
    Given the cursor to a SQLite database, check if the specified
    file hash is already in the database.
    """
    cursor.execute("SELECT 1 FROM files WHERE file_hash = ?", (file_hash,))
    return cursor.fetchone() is not None

def insert_file(cursor, file_hash):
    """
    Inserts a file into the database.
    """
    cursor.execute("INSERT INTO files (file_hash) VALUES (?)", (file_hash,))