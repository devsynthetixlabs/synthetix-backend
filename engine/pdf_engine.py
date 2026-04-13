import pdfplumber
import os

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


def get_relevant_pdf_context(question, tenant_id):
    # 1. Extract keywords from the question (e.g., "Discount Policy")
    # 2. Query the 'document_knowledge' table
    query = text("""
        SELECT content FROM document_knowledge 
        WHERE tenant_id = :tid 
        AND content ILIKE :search
        LIMIT 2
    """)
    
    with db._engine.connect() as conn:
        results = conn.execute(query, {
            "tid": tenant_id, 
            "search": f"%{question[:20]}%" # Simplistic keyword search
        }).fetchall()
        
    return "\n".join([r[0] for r in results])