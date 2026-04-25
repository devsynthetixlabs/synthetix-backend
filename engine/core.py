# engine/core.py
import os
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