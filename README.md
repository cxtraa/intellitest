# Intellitest.

### What is Intellitest?

Intellitest is an LLM application that produces bespoke exam questions for you based on your past exam documents. 

Upload past exam documents, type in a query like, "Generate an exam problem on thermodynamics", and watch high-quality questions and solutions unfold in front of you!

## Usage instructions

Currently, the application is not available on the web. Therefore, you will have to clone the repository locally. Then, run the following command from within `/intellitest`:

```bash
streamlit run app.py
```

A link will be generated where you can interface with the model. Note that you will need an OpenAI API key to use the application - you can get one by [signing up for an OpenAI account.](https://openai.com/)

### How does it work?

Intellitest runs using the OpenAI API. The language model used is `gpt-3.5-turbo`, and the embedding model used is `text-embedding-3-small`.

When you upload files, we check to see if your file already exists on our database. If so, the embeddings can be directly accessed from our vector database. Otherwise, your documents are chunked (turned into smaller, digestable pieces), and embedded into 1536-dimensional vectors.

Your query is also embedded, and the cosine similarity metric is used to find the `N_CONTEXT` most similar questions in our database. These are returned and placed into the models context window, along with some auto-generated keywords that help enrich the model query.

All of this helps GPT-3.5 produce a high-quality response that is tailored to your specific exam.
