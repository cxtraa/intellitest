"""
All system prompts for LLM pipeline.
Also includes functions for querying model.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import PyPDF2
import pprint

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils import *
from settings import *

def one_question_pipeline(client, augmented_query, temps, initial_prompt):
    """
    Produce a problem and solution on a given academic topic.

    Inputs:
    - client : the OpenAI client instance being used.
    - augmented_query : a str containing the user query and relevant context for the model.
    - temps : a dictionary mapping a part of the pipeline to the temperature to use for that part.
    """

    model = st.session_state["openai_model"]

    # System prompt for question generator (unrefined)
    unrefined_question_sysprompt = f"""
    The user has provided the following information about their level of education and desired subject: {initial_prompt}.

    You are an expert question setter for the user's desired education level and subject.

    Your job is to produce difficult exam problems at the required education level on demand when prompted by the user.

    The questions should be creative, soleveable, and logically consistent.

    You must only return the exam problem. Do not provide the solution as this could allow the student to cheat.

    Use LaTeX for all mathematical expressions.
    """

    refined_question_sysprompt = f"""
    Given the user's educational level and subject preference: {initial_prompt}, your task is to review and enhance the clarity, precision, and appropriateness of the following exam question. Assume the role of an expert in educational content creation, ensuring that the question aligns with the specified academic standards and effectively assesses the subject matter. Present the refined question using clear and concise language, suitable for inclusion in an examination.
    """

    # System prompt for answering refined question (unrefined)
    unrefined_answer_sysprompt = f"""
    The user has provided the following information about their level of education and desired subject: {initial_prompt}

    You are a specialized AI agent who is an expert at answering exam problems at the user's desired education level and desired subject.

    Given an exam problem, you must provide a detailed solution to the problem, which may contain multiple parts.

    The solutions must be clear, logical, and correct.

    Use LaTeX for all mathematical expressions.
    """

    refined_answer_sysprompt = f"""
    The user has provided the following information about their level of education and desired subject: {initial_prompt}

    As a quality control officer for exam solutions tailored to the user's educational level and subject, your role is to ensure the highest quality of the solution provided. Please review the following proposed solution and make any necessary corrections or improvements. Present the optimal solution, using LaTeX for all mathematical expressions, as if it were the final solution to be delivered to the user.
    """
    
    # Produced unrefined question
    unrefined_question_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content" : unrefined_question_sysprompt},
            {"role" : "user", "content" : augmented_query}
        ],
        temperature=temps["unrefined_question"],
    )
    unrefined_question = unrefined_question_response.choices[0].message.content

    refined_question_query = f"""
    Here is the original exam question:

    {unrefined_question}

    Review and refine this question to ensure it is clearly articulated and appropriately challenging for the user's level of education and subject area. The final version should be ready to be presented as part of an exam, ensuring it effectively tests the relevant knowledge and skills.
    """

    # Feed unrefined question into model to get refined questions
    refined_question_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content" : refined_question_sysprompt},
            {"role" : "user", "content" : refined_question_query},
        ],
        temperature = temps["refined_question"]
    )
    refined_question = refined_question_response.choices[0].message.content
    refined_question = convert_latex_format(refined_question)

    # Create a user prompt for passing in the refined question to the model with instructions
    unrefined_answer_query = f"""
    Provide a solution(s) to the following university-level engineering exam problem(s):

    {refined_question}
    """

    # Feed the refined question to the answerer to give an unrefined solution
    unrefined_answer_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content" : unrefined_answer_sysprompt},
            {"role" : "user", "content" : unrefined_answer_query}
        ],
        temperature = temps["unrefined_answer"]
    )
    unrefined_answer = unrefined_answer_response.choices[0].message.content

    refined_answer_query = f"""
    Based on the question: {refined_question}, here is the proposed solution: {unrefined_answer}

    Please review and adjust this solution to ensure it meets the highest standards of accuracy and clarity, presenting the final, polished solution only.
    """

    refined_answer_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content" : refined_answer_sysprompt},
            {"role" : "user", "content" : refined_answer_query}
        ],
        temperature=temps["refined_answer"]
    )
    refined_answer = refined_answer_response.choices[0].message.content
    refined_answer = convert_latex_format(refined_answer)

    return refined_question, refined_answer
