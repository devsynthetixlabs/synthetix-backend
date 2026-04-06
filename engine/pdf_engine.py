import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(BASE_DIR, "storage")
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_pdf_engine():
    # If storage exists, load it. If not, create it once.
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(
            DATA_DIR, 
            recursive=True, 
            exclude=["financials/*", "*.db",]
        ).load_data()
        
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    return index.as_query_engine()

def ask_policy(question):
    query_engine = get_pdf_engine()
    
    # 🤖 Neutral & Friendly Prompt
    neutral_prompt = (
        f"Context information is below. Answer the following question: {question}\n"
        "Guidelines:\n"
        "1. Provide a clear, friendly sentence.\n"
        "2. Always include the specific Date (Month, Day, Year) and the Day of the week.\n"
        "3. Avoid internal jargon or personas. Just give the information accurately."
    )
    
    response = query_engine.query(neutral_prompt)
    return str(response).strip()