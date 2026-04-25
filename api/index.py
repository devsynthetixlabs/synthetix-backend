import os
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from database.db import SessionLocal
from sqlalchemy import text
from engine.pdf_engine import extract_structured_text
from api.auth import router as auth_router
from api.auth import get_current_user # Import your dependency
from langchain_community.utilities import SQLDatabase
from engine.pdf_engine import get_relevant_pdf_context_with_rerank
from engine.core import db, llm

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://synthetix-web-app.vercel.app",
    "http://192.168.31.231:8000",
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
    # tenant_id: str

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
    return str(response.text).strip().upper()

def is_followup(question: str):
    followup_words = ["it", "this", "that", "they", "those", "them"]
    return any(word in question.lower() for word in followup_words)

# --- Logic Layer ---
def ask_synthetix_labs(question: str, history: List[Message], tenant_id: str):
    processed_question = question
    
    # 1. THE REFINER (Context Injection)
    if history and is_followup(question):
        recent_context = "\n".join([f"{msg.sender}: {msg.text}" for msg in history[-3:]])
        refine_prompt = f"Chat History:\n{recent_context}\nUser Question:\n{question}\nRewrite as a standalone question:"
        response = llm.complete(refine_prompt)
        processed_question = str(response.text).strip()

    # 2. THE DISPATCHER (Routing)
    route = classify_query(processed_question)

    print(route)

    if route == "STRATEGY":
        # Multi-Agent Workflow (CrewAI)
        print(f"🤖 Routing to CrewAI Strategy: {processed_question}")
        # Assuming you wrapped your Crew logic in a function called run_sales_strategy
        from engine.strategy_engine import ask_strategy
        return ask_strategy(processed_question, tenant_id)

    elif route == "SQL":
        # Single-Agent / Direct Tool Access
        print(f"📊 Routing to SQL Engine: {processed_question}")
        from engine.sql_engine import ask_cfo
        result = ask_cfo(processed_question, tenant_id)
        # Ensure we return only the string 'answer' from the dict
        return result['answer'] if isinstance(result, dict) else result
        
    else:
        # Document Search
        print(f"📄 Routing to PDF Engine: {processed_question}")
        return ask_synthetix_labs_self_rag(processed_question, tenant_id)

def ask_synthetix_labs_self_rag(question, tenant_id):
    # 1. RETRIEVE
    context = get_relevant_pdf_context_with_rerank(question, tenant_id)
    
    # 2. REFLECT: Is the context relevant?
    relevance_prompt = f"Given this question: '{question}', is the following text relevant? Answer only 'YES' or 'NO'. \nText: {context}"
    is_relevant = llm.complete(relevance_prompt).text # Fast, cheap model call

    if "NO" in is_relevant.upper():
        # Fallback: Maybe search the database again with different keywords 
        # or tell the user the documents aren't helpful.
        return "I found some documents, but they don't seem to address your specific question."

    # 3. GENERATE
    generation_prompt = f"""
    You are a concise office assistant. Use ONLY the provided context to answer the question.
    If the information is not in the context, say you don't know. 
    Do not provide general historical or cultural background unless specifically asked.

    CONTEXT:
    {context}

    QUESTION: 
    {question}

    ANSWER:
    """
    draft_answer = llm.complete(generation_prompt).text

    # 4. REFLECT: Does the answer hallucinate?
    hallucination_check = f"Does the answer '{draft_answer}' contain info NOT present in the context: '{context}'? Answer 'SAFE' or 'HALLUCINATION'."
    check_result = llm.complete(hallucination_check).text

    if "HALLUCINATION" in check_result.upper():
        return "I'm sorry, I can't find a factual basis for that answer in your uploaded documents."

    return draft_answer

@app.post("/ask")
def handle_query(request: QueryRequest, user: dict = Depends(get_current_user)):
    # 1. Extract context from the verified token, NOT the request body
    # This prevents 'Tenant Spoofing' (User A asking for User B's data)
    tenant_id = user.get("tenant_id")
    
    # 2. Pass the verified tenant_id into your AI engine
    answer = ask_synthetix_labs(
        request.question, 
        request.history, 
        tenant_id
    )
    
    return {"answer": answer}

@app.post("/upload-knowledge")
async def upload_doc(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    # 1. Securely get the tenant_id from the token, not the request body
    tenant_id = user.get("tenant_id")
    
    # 2. Save file temporarily
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 3. Parse intelligently
        full_text = extract_structured_text(file_path)

        # 4. Save to Neon Database
        query = text("""
            INSERT INTO document_knowledge (tenant_id, file_name, content)
            VALUES (:tid, :fname, :cont)
        """)
        
        with db._engine.connect() as conn:
            conn.execute(query, {
                "tid": tenant_id,
                "fname": file.filename,
                "cont": full_text
            })
            conn.commit() # Important for non-autocommit engines

        return {
            "status": "success", 
            "message": f"Successfully learned from {file.filename} for tenant {tenant_id}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    finally:
        # 5. Cleanup temp file regardless of success or failure
        if os.path.exists(file_path):
            os.remove(file_path)

app.include_router(auth_router)
