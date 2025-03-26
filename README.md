# Gaia Sky AI assistant

This project implements an AI assistant for Gaia Sky. It consists of a RAG setup that scrapes the Gaia Sky documentation and home page, transforms it to text, does the embedding to vectors, and persists them to Chroma DB. Then, a chatbot is provided to query any Ollama model with relevant context.

This project is described in the following blog post:

https://tonisagrista.com/blog/2025/gaiasky-ai-assistant

## Running

In order to run it, clone the repository and set up the virtual environment with `pipenv`:

```bash
# Install dependencies
pipenv install
# Enter the virtual environment
pipenv shell
```

Then, run the `gsai.py` script. Use the `--scrape` flag the first time so that the websites are scraped and the database is populated. This is not required the subsequent times.

```bash
python gsai.py --scrape
```

