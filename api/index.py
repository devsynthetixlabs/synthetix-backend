import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://synthetix-web-app.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Message(BaseModel):
    id: str
    sender: str
    text: str

class QueryRequest(BaseModel):
    question: str
    history: List[Message]

# --- Classification Logic ---
def classify_query(question: str):
    prompt = f"""
    Classify the user query into EXACTLY one of these three categories:

    1. PDF: Use this for internal company information, HR policies, 
       HOLIDAY lists, leave policies, office timings, or any document-based facts.
       (Example: "When is Diwali?", "Is Friday a holiday?", "Sick leave policy")

    2. SQL: Use this for specific NUMERIC sales data, invoices, 
       revenue, customer names, or item quantities from the database.
       (Example: "Total sales in 2023", "Who is our top client?")

    3. STRATEGY: Use ONLY if the user is asking for advice, a 3-step plan, 
       marketing suggestions, growth ideas, or business consulting.
       (Example: "Suggest a plan to grow PRALCKA", "Marketing strategy for 2024")

    Question: {question}

    Answer ONLY with the word: PDF, SQL, or STRATEGY
    """

    response = llm.complete(prompt)
    return response.text.strip().upper()

def is_followup(question: str):
    followup_words = ["it", "this", "that", "they", "those", "them"]
    return any(word in question.lower() for word in followup_words)

# --- Logic Layer ---
def ask_synthetix_labs(question: str, history: List[Message]):
    processed_question = question
    
    # 1. THE REFINER (Context Injection)
    if history and is_followup(question):
        recent_context = "\n".join([f"{msg.sender}: {msg.text}" for msg in history[-3:]])
        refine_prompt = f"Chat History:\n{recent_context}\nUser Question:\n{question}\nRewrite as a standalone question:"
        response = llm.complete(refine_prompt)
        processed_question = response.text.strip()

    # 2. THE DISPATCHER (Routing)
    route = classify_query(processed_question)

    print(route)

    if route == "STRATEGY":
        # Multi-Agent Workflow (CrewAI)
        print(f"🤖 Routing to CrewAI Strategy: {processed_question}")
        # Assuming you wrapped your Crew logic in a function called run_sales_strategy
        from engine.strategy_engine import ask_strategy
        return ask_strategy(processed_question)

    elif route == "SQL":
        # Single-Agent / Direct Tool Access
        print(f"📊 Routing to SQL Engine: {processed_question}")
        from engine.sql_engine import ask_cfo
        result = ask_cfo(processed_question)
        # Ensure we return only the string 'answer' from the dict
        return result['answer'] if isinstance(result, dict) else result
        
    else:
        # Document Search
        print(f"📄 Routing to PDF Engine: {processed_question}")
        from engine.pdf_engine import ask_policy
        return ask_policy(processed_question)

@app.post("/ask")
def handle_query(request: QueryRequest):
    answer = ask_synthetix_labs(request.question, request.history)
    return {"answer": answer}