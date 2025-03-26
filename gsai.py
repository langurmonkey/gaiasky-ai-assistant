#! python
import os
# Set user agent enviornment variable
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ["USER_AGENT"] = user_agent

import requests, ollama, argparse, readline
from termcolor import colored
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# MAIN PARAMETERS =======================================

# REQUEST HEADER
headers = {
    "User-Agent": user_agent
}

# EMBED MODEL
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
# embed_model = "BAAI/bge-base-en-v1.5"
# embed_model = "HIT-TMG/KaLM-embedding-multilingual-mini-v1"
# embed_model = "Linq-AI-Research/Linq-Embed-Mistral"

# SYSTEM PROMPT
system_prompt = (
    "You are a customer support agent, helping users by following directives and answering questions. "
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise and to the point. "
    "Context: {context}"
)

# ======================================================

def get_all_doc_links(base_url):
    """Finds all internal links within the documentation site."""
    visited = set()
    to_visit = {base_url}

    while to_visit:
        url = to_visit.pop()
        print(f"Scraping {colored(url, 'blue')}")
        if url in visited:
            continue
        visited.add(url)

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(base_url, link["href"])
            parsed = urlparse(full_url)

            # Ignore anchors (#...) and non-HTML files
            if parsed.fragment or not parsed.path.endswith((".html", "/")):
                continue

            if base_url in full_url and full_url not in visited:
                to_visit.add(full_url)

    return visited

def clean_text(text):
    """Cleans and normalizes extracted text."""
    text = text.strip()
    text = ' '.join(text.split())  # Remove excessive whitespace
    return text if text else None

def extract_text_from_page(url):
    """Extracts meaningful text from a given URL using html2text."""
    import html2text
    try:
        response = requests.get(url)
        h = html2text.HTML2Text()
        h.ignore_links = False  # Set this to True to ignore links
        return h.handle(response.text)
    except:
        return None

def extract_text_from_page_local(url):
    """Extracts meaningful text from a given URL while stripping unwanted UI elements."""
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements (menus, footers, sidebars, etc.)
    for tag in soup.find_all(["nav", "header", "footer", "aside", "form", "script", "style"]):
        tag.decompose()

    # Extract text from relevant content elements
    content_blocks = set()
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "article", "code"]):
        text = clean_text(tag.get_text())
        if text:
            content_blocks.add(text)  # Avoid duplicate blocks

    # Extract image descriptions
    for img in soup.find_all("img"):
        alt_text = img.get("alt") or "Image without description"
        content_blocks.add(f"[Image: {alt_text}]")

    # Standardize document structure
    structured_text = "\n".join(sorted(content_blocks))

    return structured_text if structured_text else None

def scrape_urls(base_urls, local_scraper=False):
    """Scrape and extract text content from multiple URLs."""
    all_text = ""
    
    for base_url in base_urls:
        all_links = get_all_doc_links(base_url)
        print(f"Found {len(all_links)} pages to scrape...")
        print(f"Extracting text from pages...")
        for url in all_links:
            page_text = extract_text_from_page(url) if not local_scraper else extract_text_from_page_local(url)
            if page_text:
                all_text += f"\n\n### {url}\n{page_text}"

    return all_text

def store_embeddings(texts, db_path):
    """Tokenizes, embeds, and stores the texts in a ChromaDB vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([texts])
    if not documents:
        raise ValueError("No documents found to embed. Check the scraper output.")
    
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=db_path)
    return vector_store

def query_with_ollama(vector_store, model_name, num_results, temp):
    """Uses an Ollama model to retrieve and answer user queries based on stored embeddings."""
    retriever = vector_store.as_retriever(search_kwargs={"k": num_results})
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(OllamaLLM(model=model_name, temperature=temp, stream=True), prompt)
    chain = create_retrieval_chain(retriever, qa_chain)
    
    while True:
        query = input(colored("Ask a question (type 'exit' to quit): ", "yellow", attrs=["bold"]))
        if query.lower() in ["exit", "quit", "bye"]:
            break

        for chunk in chain.stream({"input": query}):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
    
        print("\n")

def chatbot(vector_store, num_results, temp):
    # List available models
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
    except:
        print(f"{colored('ERROR', 'red')}: Ollama service is not running.")
        exit(1)

    # Print the available models with index numbers
    print("Available models:")
    for i, name in enumerate(model_names):
        print(f" [{colored(i, 'green')}] {name}")

    # Loop until a valid selection is made
    while True:
        choice = input(f"\nSelect model (default {colored('0', 'green')}): ").strip()

        if choice == "":  # Default to 0 if empty input
            llm_model = model_names[0]
            break

        if choice.isdigit():  # Check if input is a number
            index = int(choice)
            if 0 <= index < len(model_names):  # Check if index is valid
                llm_model = model_names[index]
                break

        print("Invalid selection. Please enter a valid number.")

    print(f"LLM model: {colored(llm_model, 'yellow', attrs=['bold'])}")
    print(f"Embed model: {colored(embed_model, 'yellow', attrs=['bold'])}")

    print("Connecting to Ollama for queries...")
    query_with_ollama(vector_store, llm_model, num_results, temp)
    

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive integer")
    return ivalue

def bounded_float(value):
    fvalue = float(value)
    if not (0 <= fvalue <= 1):
        raise argparse.ArgumentTypeError(f"{value} is out of range [0,1]")
    return fvalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Gaia Sky AI assistant",
                                     description="AI assistant that provides answers to Gaia Sky-related questions by searching the docs for you. First, run with --scrape to populate the database. Then, run without arguments to start the chatbot.",
                                     epilog="Written by Toni Sagrista, 2025.")
    parser.add_argument("-k", "--nresults", type=positive_int, default=6, help="Number of results to use for context, as a positive integer (>0).")
    parser.add_argument("-t", "--temp", type=bounded_float, default=0.3, help="The temperature controls the 'creativity' of the model. It is in [0,1]. Lower values (0.0-0.3) make the model more correct and deterministic. Higher values (0.7-1.0) offer more creative and varied responses. We want correctness, so we use a low value.")

    parser.add_argument("-s", "--scrape", action="store_true", help="Only scrape websites and update embeddings.")
    parser.add_argument("-l", "--local-scraper", action="store_true", help="Use local scraper (less accurate!) instead of the one based on newspaper3k. Only useful when --scrape is used.")
    args = parser.parse_args()

    urls = ["https://gaia.ari.uni-heidelberg.de/gaiasky/docs/master/",
            "https://gaiasky.space"]
    db_path = "chroma_db"

    if args.local_scraper and not args.scrape:
        print(f"{color('WARNING', 'orange')}: --local-scraper has no effect without --scrape!\n")
    
    if args.scrape:
        print("Starting scraping...")
        doc_text = scrape_urls(urls, args.local_scraper)
        print("Storing embeddings in ChromaDB...")
        vector_store = store_embeddings(doc_text, db_path)
        print("Done! Run the script without arguments to start the chatbot.")
    else:
        print(colored("WELCOME TO THE GAIA SKY AI ASSISTANT!", "blue", attrs=["bold"]))

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Start chatbot.
        chatbot(vector_store, args.nresults, args.temp)
        print("Bye!")
