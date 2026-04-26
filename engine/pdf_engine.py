import pdfplumber
import os
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from engine.core import db, llm, get_embedding

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

def re_rank_context(question, candidate_objects):
    """
    Updated to handle dictionaries containing metadata.
    candidate_objects: List[dict] -> [{"content": "...", "file_name": "..."}]
    """
    try:
        if not candidate_objects:
            return []

        # 1. Extract only the text for the scoring model
        # This prevents the "expected str instance, dict found" error
        texts_to_score = [obj["content"] for obj in candidate_objects]

        # 2. Perform your re-ranking logic (Cross-Encoder / LLM Scoring)
        # Example using a simple cross-encoder:
        # scores = cross_encoder.predict([(question, text) for text in texts_to_score])
        
        # 3. Filter/Sort the original objects based on scores
        # For now, let's just ensure we return the objects themselves
        # so the metadata (filenames) stays attached.
        
        # If you are using a simple filter:
        relevant_objects = [
            obj for obj in candidate_objects 
            if len(obj["content"]) > 0 # Replace with your actual scoring logic
        ]

        return relevant_objects

    except Exception as e:
        print(f"❌ Re-ranker Error: {e}")
        # Fallback: Return original objects so the system doesn't crash
        return candidate_objects

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

def get_hybrid_pdf_context(question, tenant_id):
    try:
        # 1. Embed the question
        question_vector = get_embedding(question)
        print(f"Vector generated. Size: {len(question_vector)}")

        # 2. Execute SQL
        query = text("""
            WITH vector_matches AS (
                SELECT id, content, file_name, created_at
                FROM document_knowledge
                WHERE tenant_id = :tid
                ORDER BY embedding <=> :vector
                LIMIT 5
            ),
            keyword_matches AS (
                SELECT id, content, file_name, created_at
                FROM document_knowledge
                WHERE tenant_id = :tid 
                AND content ILIKE :keyword
                LIMIT 5
            )
            SELECT DISTINCT ON (id) content, file_name, created_at 
            FROM (
                SELECT * FROM vector_matches
                UNION ALL
                SELECT * FROM keyword_matches
            ) combined_results;
        """)

        with db._engine.connect() as conn:
            results = conn.execute(query, {
                "tid": tenant_id,
                "vector": str(question_vector),
                "keyword": f"%{question}%"
            }).fetchall()
            
        print(f"SQL Results Found: {len(results)}")

        # 3. Format results
        docs = [{"content": r[0], "file_name": r[1], "created_at": r[2]} for r in results]
        
        # 4. RE-RANK (Check if re_rank_context is working!)
        if docs:
            print("Passing docs to re-ranker...")
            final_docs = re_rank_context(question, docs)
            print(f"Final docs after re-rank: {len(final_docs)}")
            return final_docs
        
        return []

    except Exception as e:
        print(f"❌ HYBRID ERROR: {str(e)}")
        return []