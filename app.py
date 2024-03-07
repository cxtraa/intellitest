"""
Application frontend.
"""

#%% Import modules
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

from utils import *
from prompts import *
from settings import *

#%% Setup session state
st.title("Intellitest")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = SentenceTransformer('all-MiniLM-L6-v2')

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if 'init_done' not in st.session_state:
    st.session_state['init_done'] = False
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = False
if 'initial_prompt' not in st.session_state:
    st.session_state['initial_prompt'] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state['uploaded_files'] and st.session_state['initial_prompt']:
    st.session_state['init_done'] = True

#%% Get initial prompt and user documents
if not st.session_state['initial_prompt']:
    with st.container():
        st.markdown(WELCOME)
    initial_prompt = st.chat_input("Enter education level and subject")    
    st.session_state['initial_prompt'] = initial_prompt
    if initial_prompt:
        st.rerun()

if not st.session_state['uploaded_files']:
    uploaded_files = st.file_uploader("Upload your exam documents here.", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        all_chunks = []
        for uploaded_file in uploaded_files:
            raw_text = extract_text_from_pdf(uploaded_file)
            chunks = clean_and_chunk_text(raw_text)
            all_chunks += chunks
        all_chunks_embeddings = get_embeddings(all_chunks, model)
        st.session_state['all_chunks_embeddings'] = all_chunks_embeddings
        st.session_state['all_chunks'] = all_chunks
        st.session_state['uploaded_files'] = True
        st.rerun()

#%% Main querying section
if st.session_state['init_done']:
    with st.container():
        st.markdown(START_QUERYING)
    prompt = st.chat_input("Query")
    if prompt:
        st.session_state.messages.append({
                "role" : "user",
                "content" : prompt,
            })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        query_embedding = get_embeddings([prompt], model)[0]

        context = retrieve_context(
            query_embedding=query_embedding,
            chunk_embeddings=st.session_state['all_chunks_embeddings'],
            chunks=st.session_state['all_chunks'],
            N=N_CONTEXT,
        )
        context = "\n".join(context)

        augmented_query = f"""
        Based on the following examples, generate a new and unique exam problem. The examples are for inspiration only and should not be used directly in the new question: {context}

        User query for generating questions: {prompt}
        """

        # For debugging only
        # with st.chat_message("user"):
        #     st.markdown(augmented_query)
        
        problem, solution = one_question_pipeline(
            client=client,
            initial_prompt=st.session_state['initial_prompt'],
            augmented_query=augmented_query,
            temps={
                "unrefined_question" : 0.3,
                "refined_question" : 0.3,
                "unrefined_answer" : 0.3,
                "refined_answer" : 0.3,
            }
        )
        with st.chat_message("assistant"):
            response_problem = st.markdown(problem)
        with st.chat_message("assistant"):
            response_solution = st.markdown(solution)

        st.session_state.messages.append({
            "role": "assistant",
            "content": problem,
            })
        st.session_state.messages.append({
            "role": "assistant",
            "content": solution,
            })

