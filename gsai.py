import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

llm_model = "hermes3:latest"

def get_all_doc_links(base_url):
    """Finds all internal links within the documentation site."""
    visited = set()
    to_visit = {base_url}

    while to_visit:
        url = to_visit.pop()
        print(f"Scrapping {url}")
        if url in visited:
            continue
        visited.add(url)

        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GaiaSkyScraper/1.0)"})
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(base_url, link["href"])
            if base_url in full_url and full_url not in visited:
                to_visit.add(full_url)

    return visited

def scrape_gaia_sky_docs(base_url):
    """Scrape and extract text from all documentation pages."""
    all_links = get_all_doc_links(base_url)
    print(f"Found {len(all_links)} pages to scrape...")

    all_text = ""
    for url in all_links:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; GaiaSkyScraper/1.0)"})
        if response.status_code != 200:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = '\n'.join([p.get_text() for p in soup.find_all("p")])
        all_text += f"\n\n### {url}\n{page_text}"

    return all_text

def store_embeddings(texts, db_path="chroma_db"):
    """Tokenizes, embeds, and stores the texts in a ChromaDB vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.create_documents([texts])
    if not documents:
        raise ValueError("No documents found to embed. Check the scraper output.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=db_path)
    return vector_store

def query_with_ollama(vector_store, model_name=llm_model):
    """Uses an Ollama model to retrieve and answer user queries based on stored embeddings."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Avoid exceeding stored docs
    qa_chain = RetrievalQA.from_chain_type(llm=OllamaLLM(model=model_name), retriever=retriever)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = qa_chain.invoke({"query": query})
        print("Answer:", answer)

if __name__ == "__main__":
    base_url = "https://gaia.ari.uni-heidelberg.de/gaiasky/docs/master/"
    print("Scraping Gaia Sky documentation...")
    doc_text = scrape_gaia_sky_docs(base_url)
    
    print("Storing embeddings in ChromaDB...")
    vector_store = store_embeddings(doc_text)
    
    print("Connecting to Ollama for queries...")
    query_with_ollama(vector_store)

