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
from engine.pdf_engine import get_hybrid_pdf_context
from engine.core import db, llm, get_embedding

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.31.231:8000",
        "https://synthetix-web-app.vercel.app",
        "https://synthetix-backend.vercel.app",
    ],
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
       (Example: "When is Diwali?", "Is Friday a holiday?", "sick leave policy")

    2. SQL: Use this for specific NUMERIC sales data, invoices, 
       revenue, customer names, item quantities, most sold items,
       product orders, top products, or item-level data from the database.
       (Example: "Total sales in 2023", "What item was ordered the most?", 
       "Top 5 products", "Which product sold most", "How many units")

    3. STRATEGY: Use ONLY if the user is asking for advice, a 3-step plan, 
       marketing suggestions, growth ideas, or business consulting.
       (Example: "Suggest a plan to grow PRALCKA", "Marketing strategy for 2024")

    Question: {question}

    Answer ONLY with the word: PDF, SQL, or STRATEGY
    """

    response = llm.complete(prompt)
    result = str(response.text).strip().upper()
    print(f"🔍 Classification result: {result}")
    return result

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
        print(f"📊 SQL Result: {result}")
        # Ensure we return only the string 'answer' from the dict
        return result['answer'] if isinstance(result, dict) else result
        
    else:
        # Document Search
        print(f"📄 Routing to PDF Engine: {processed_question}")
        return ask_synthetix_labs_self_rag(processed_question, tenant_id)

def ask_synthetix_labs_self_rag(question, tenant_id):
    # 1. RETRIEVE
    raw_context_data = get_hybrid_pdf_context(question, tenant_id)
    
    if not raw_context_data:
        return "I couldn't find any documents in your knowledge base."

    # 2. FORMAT (Ensuring metadata is clear for the LLM)
    formatted_context = ""
    for idx, doc in enumerate(raw_context_data):
        formatted_context += (
            f"\n### DOCUMENT {idx+1} ###\n"
            f"FILE: {doc.get('file_name')}\n"
            f"DATE: {doc.get('created_at')}\n"
            f"CONTENT: {doc.get('content')}\n"
        )

    # 3. RELEVANCE CHECK (The 'Gatekeeper')
    relevance_prompt = f"""
    Evaluate if this context helps answer: "{question}"
    Look for keywords, dates, or lists in the 'CONTENT' sections.
    Answer only YES or NO.
    
    CONTEXT:
    {formatted_context}
    """
    
    # Use .strip() and .upper() to avoid "I couldn't process" errors on malformed strings
    is_relevant = llm.complete(relevance_prompt).text.strip().upper()

    # FAIL-SAFE: If the model is being too picky, we force a YES if the list isn't empty
    if "NO" in is_relevant and len(raw_context_data) > 0:
        # Log this so you know the filter was too strict
        print(f"⚠️ Relevance filter was too strict for query: {question}")
        is_relevant = "YES"

    if "YES" not in is_relevant:
        return "The documents I found don't seem to contain a specific answer for you."

    # 4. GENERATE (With Citation & Conflict Logic)
    generation_prompt = f"""
    You are the Synthetix Labs HR Assistant. Your goal is to provide clear, 
    structured, and professional information based ONLY on the provided context.

    USER QUESTION: "{question}"
    CONTEXT: {formatted_context}

    STYLE GUIDELINES:
    1. **Direct Answer**: Start with a summary sentence. Do NOT start with "Yes" or "No".
    2. **Formatting**: Use bullet points for lists, dates, or multiple items to ensure the answer is scannable.
    3. **Tone**: Professional yet accessible. Avoid overly dense legalese like "constitutes" unless necessary.
    4. **Citations**: Mention the source file name in **bold** at the very end of the response.
    5. **Constraint**: If the context doesn't contain the answer, state that you don't have that information on file.

    STRUCTURE:
    ## [Heading for the Topic]
    - A 1-2 sentence high-level summary.
    - **Key Details**: Use a bulleted list for specifics (dates, rules, or requirements).
    - **Note**: Mention any specific conditions (like the Sunday-Monday holiday rule).
    
    Source: **[Source File Name]**
    """
    
    draft_answer= llm.complete(generation_prompt).text

    # 4. REFLECT: Does the answer hallucinate?
    hallucination_check = f"""
    Compare the Answer to the Context below.
    Question: {question}
    Context: {raw_context_data}
    Answer: {draft_answer}

    Is the Answer supported by the Context? 
    - Answer 'SAFE' if the dates, names, and facts in the Answer are present in the Context.
    - Answer 'HALLUCINATION' only if the Answer makes up information NOT found in the Context.
    - Minor rephrasing is fine.

    Answer ONLY 'SAFE' or 'HALLUCINATION':
    """
    check_result = llm.complete(hallucination_check).text.strip()

    if "HALLUCINATION" in check_result.upper():
        # Optimization: If the answer is about a holiday and matches a date in the context,
        # it's likely safe. We can refine this check later.
        return "I'm sorry, I can't find a factual basis for that answer in your uploaded documents."

    return {
    "answer": draft_answer,
    "sources": [doc.get('file_name') for doc in raw_context_data],
    "status": "success"
    }

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
    tenant_id = user.get("tenant_id")
    file_path = f"temp_{file.filename}"
    
    try:
        # 1. Save file temporarily
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 2. Parse intelligently
        full_text = extract_structured_text(file_path)

        # 3. Generate 768-dim Gemini Embedding
        # This now calls the google-genai logic we set up in core.py
        vector_representation = get_embedding(full_text)

        if not vector_representation:
            raise ValueError("Failed to generate embedding from Gemini.")

        # 4. Save to Neon with correct formatting
        query = text("""
            INSERT INTO document_knowledge (tenant_id, file_name, content, embedding)
            VALUES (:tid, :fname, :cont, :vec)
        """)
        
        with db._engine.connect() as conn:
            conn.execute(query, {
                "tid": tenant_id,
                "fname": file.filename,
                "cont": full_text,
                # pgvector expects a string representation of a list: "[0.1, 0.2, ...]"
                "vec": str(vector_representation) 
            })
            conn.commit()

        return {
            "status": "success", 
            "message": f"Successfully indexed {file.filename} with Gemini 768-dim vectors."
        }

    except Exception as e:
        print(f"Upload error: {e}")
        # Detailed error for debugging during the migration
        return {"status": "error", "message": f"Processing failed: {str(e)}"}
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

app.include_router(auth_router)
