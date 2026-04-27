# engine/core.py
import os
import requests
from google import genai
from google.genai import types 
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
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1beta'} # Changed from v1 to v1beta
)

def get_embedding(text):
    if not text:
        return None
        
    clean_text = str(text).replace("\n", " ")
    
    try:
        # gemini-embedding-2 is natively multimodal and handles PDFs/Text
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=clean_text,
            config=types.EmbedContentConfig(
                # Ensure this matches your 768-dim DB column
                output_dimensionality=768,
            )
        )
        
        if result.embeddings:
            return result.embeddings[0].values
        return None
        
    except Exception as e:
        print(f"Embedding Error: {e}")
        # If gemini-embedding-2 is still finicky, 'text-embedding-004' 
        # may still work ONLY if you switch back to 'v1'
        raise Exception(f"Google API Error: {e}")

# Initialize singletons
db = get_db()
llm = get_llm()