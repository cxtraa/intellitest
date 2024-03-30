"""
Constants and settings.
"""

# Number of chunks to use for context in each query
N_CONTEXT = 2

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Language model
LANGUAGE_MODEL = "gpt-3.5-turbo"

# Path to vector database and mapping from embeddings to text chunks
FILES_DB_PATH = "./files.db"
VECTOR_DB_PATH = "./chunks.ann"
EMBEDDINGS_TO_TEXT_PATH = "./chunks_mapping.pkl"

# Temperature for prompts
TEMPS = {
    "keywords" : 0.1,
    "question" : 0.8,
    "answer" : 0.5,
}

# Welcome message
WELCOME = """
Welcome to Intellitest.

Intellitest is a web application designed for Cambridge Engineering students. It generates exam questions when prompted, using context from Tripos exam questions uploaded to the system. 
Please note that this system is designed to work with Cambridge Engineering papers. It may not function correctly for other exams.
"""

START_QUERYING = """
An example query to get you started: "Generate an exam problem on thermodynamics."
"""

CLEAN_INPUT_SYSPROMPT = """
Assistant is a large language model trained by OpenAI.
"""

KEYWORDS_SYSPROMPT = """
You are a helpful large language model.
You output 10 similar words to the keywords in a sentence.
"""

QUESTION_SYSPROMPT = """
You are an exam question setter for the Cambridge Engineering Tripos.
You must produce difficult exam problems that require deep problem-solving skills.
The questions must be logical and should have concrete answers, either numerical or algebraic.
The questions must consistent of text only.
You must only give the exam problem and nothing else. 
Use LaTeX for all mathematical expressions.
"""

ANSWER_SYSPROMPT = """
You are an expert at producing solutions to engineering problems for the Cambridge Engineering Tripos.
The solutions you produce are step-by-step, and you check each step logically to make sure the whole argument is logical.
You do not miss out any steps between the question and getting to the solution.
You are not afraid to give long responses if necessary to fully answer the problem.
You use LaTeX for all mathematical expressions.
"""