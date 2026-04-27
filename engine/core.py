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
    if not url:
        raise ValueError("DATABASE_URL not found in environment variables")
        
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    # Advanced engine arguments to prevent "SSL connection closed" errors on Vercel
    engine_args = {
        "connect_args": {"sslmode": "require"},
        "pool_pre_ping": True,    # Checks if connection is alive before using it
        "pool_recycle": 30,       # Recycles connections every 30 seconds
        "pool_size": 5,           # Keeps a small pool for serverless efficiency
        "max_overflow": 0
    }
    
    return SQLDatabase.from_uri(url, engine_args=engine_args)

# --- LLM SETUP ---
def get_llm():
    return Groq(
        model="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY")
    )

# --- EMBEDDING SETUP (Hugging Face API) ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text):
    if not HF_TOKEN:
        print("Warning: HF_TOKEN is missing!")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    
    # Error handling for the API call
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
        
    return response.json()

# Initialize singletons
db = get_db()
llm = get_llm()