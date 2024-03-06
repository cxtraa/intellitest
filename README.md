# intellitest v1.0
An LLM application that generates exam questions based on uploaded documents. Supports customisable subjects and educational levels.

### Features
- Upload several exam documents (PDF)
- Specify subject (e.g. math, physics, chemistry, engineering, medicine...)
- Specify education level (e.g. elementary school, high school, undergraduate, graduate...)
- Generate bespoke exam questions that match style of uploaded documents
- Generate accurate solutions to the problems

### Implementation
- UI built using Streamlit
- OpenAI API used for making LLM requests
- Prompt chaining utilised to produce accurate problems and solutions (unrefined question -> refined question -> unrefined answer -> refined answer).
- Retrieval augmented generation (RAG) used to embed user documents into model. The model takes the top N relevant chunks (paragraphs) from the text to support its problem/solution.
