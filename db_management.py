"""
Utility functions for working with databases.

Main databases:
- files.db : stores file hashes to check if file uploaded already exists.
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
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Set, Iterable

from constants import *

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