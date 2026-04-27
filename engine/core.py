# engine/core.py
import os
import requests
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from langchain_community.utilities import SQLDatabase

load_dotenv()

# --- DATABASE SETUP ---
def get_db():
    url = os.getenv("DATABASE_URL")
    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    # Return the SQLDatabase instance
    return SQLDatabase.from_uri(
        url, 
        engine_args={"connect_args": {"sslmode": "require"}}
    )

# --- LLM SETUP ---
def get_llm():
    return Groq(
        model="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )

# Initialize singletons to be reused across the app
db = get_db()
llm = get_llm()

# Get a free API token from huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    # Returns the same 384-dim vector as your local model!
    return response.json()
