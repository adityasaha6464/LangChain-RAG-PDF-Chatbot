from dotenv import load_dotenv
import os
import time
import logging
from types import SimpleNamespace

# Exceptions from Google/genai wrapper
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

# Load API key
load_dotenv()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Prefer Groq if key provided, otherwise fall back to Gemini (if its key is present)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GROQ_API_KEY:
    print("Using Groq API (from GROQ_API_KEY)")
    use_groq = True
else:
    use_groq = False
    if GOOGLE_API_KEY:
        print("Using Google Gemini (from GOOGLE_API_KEY)")
    else:
        print("Warning: No LLM API key set. Set GROQ_API_KEY or GOOGLE_API_KEY in your environment or .env file.")

llm = None
if not use_groq and GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)


def call_groq_api(prompt, api_key, timeout=20):
    """Call Groq's completion endpoint. Returns text result or raises."""
    url = "https://api.groq.dev/v1/models/groq-1.0/complete"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 512,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Try common fields
    if isinstance(data, dict):
        # Adapt to possible response shapes
        if "text" in data:
            return data["text"]
        if "output" in data and isinstance(data["output"], list):
            # join text pieces
            return "\n".join([str(o.get("text", o)) if isinstance(o, dict) else str(o) for o in data["output"]])
        # fallback to full json
        return str(data)
    return str(data)

print("PDF Chatbot Ready!")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask Question: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final prompt
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    # Get AI response with retry + fallback
    def get_response_with_fallback(prompt, retries=2, backoff=2):
        attempt = 0
        while True:
            try:
                if use_groq and GROQ_API_KEY:
                    text = call_groq_api(prompt, GROQ_API_KEY)
                    return SimpleNamespace(content=text)
                elif llm is not None:
                    return llm.invoke(prompt)
                else:
                    return SimpleNamespace(content=(
                        "No LLM configured. Set GROQ_API_KEY or GOOGLE_API_KEY in your environment."
                    ))
            except requests.HTTPError as e:
                attempt += 1
                logging.warning("Groq HTTP error (attempt %s): %s", attempt, e)
                if attempt > retries:
                    return SimpleNamespace(content=(
                        "Sorry — the LLM service is temporarily unavailable or returned an error. Please try again later."
                    ))
                time.sleep(backoff ** attempt)
            except Exception as e:
                attempt += 1
                logging.exception("Error calling LLM (attempt %s): %s", attempt, e)
                if attempt > retries:
                    return SimpleNamespace(content=(
                        "An unexpected error occurred while generating the response. Please try again later."
                    ))
                time.sleep(backoff ** attempt)

    response = get_response_with_fallback(prompt)

    print("\nAI Answer:\n")
    print(response.content)
    print("\n" + "-"*50 + "\n")