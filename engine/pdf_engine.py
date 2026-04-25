import pdfplumber
import os
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from engine.core import db, llm

def extract_structured_text(pdf_path):
    structured_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            structured_text += f"\n--- PAGE {i+1} ---\n"
            
            # Extract tables first
            tables = page.extract_tables()
            for table in tables:
                structured_text += "\n### TABLE DATA ###\n"
                for row in table:
                    # Filter out None values and join with pipes for Markdown style
                    structured_text += "| " + " | ".join([str(x) if x else "" for x in row]) + " |\n"
            
            # Extract plain text
            text = page.extract_text()
            if text:
                structured_text += f"\n### TEXT CONTENT ###\n{text}\n"
                
    return structured_text

def re_rank_context(question, documents):
    if not documents:
        return ""

    # Prepare the list of docs for the LLM to rank
    doc_list = "\n".join([f"ID {i}: {doc}" for i, doc in enumerate(documents)])

    re_rank_prompt = f"""
    You are a search ranker. Given the User Question and a list of Document Snippets, 
    re-order the Snippets from most relevant to least relevant.
    
    User Question: {question}
    Snippets:
    {doc_list}
    
    Output ONLY the IDs of the top 2 snippets in order of relevance, separated by commas.
    Example Output: 2, 0
    """
    
    try:
        # Use your Groq LLM to pick the best ones
        response = llm.complete(re_rank_prompt).text.strip()
        best_ids = [int(i.strip()) for i in response.split(",")]
        
        # Reconstruct the context using only the top-ranked docs
        reranked_context = "\n".join([documents[i] for i in best_ids if i < len(documents)])
        return reranked_context
    except:
        # Fallback to original order if re-ranking fails
        return "\n".join(documents[:2])

def get_relevant_pdf_context_with_rerank(question, tenant_id):
    # 1. Fetch MORE documents than before (e.g., LIMIT 5)
    query = text("""
        SELECT content FROM document_knowledge 
        WHERE tenant_id = :tid 
        LIMIT 5
    """)
    
    with db._engine.connect() as conn:
        results = conn.execute(query, {"tid": tenant_id}).fetchall()
        docs = [r[0] for r in results]
        
    # 2. Pass them through the Re-ranker
    return re_rank_context(question, docs)