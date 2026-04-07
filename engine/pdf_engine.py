import os
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ✨ FIX 1: Initialize the client clearly
# The 2026 SDK usually defaults to v1. We want to ensure we're on the right path.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_context_from_pdfs():
    text_content = ""
    if not os.path.exists(DATA_DIR):
        return "No documents found."

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    return text_content

def ask_policy(question):
    context = get_context_from_pdfs()
    
    prompt = f"Using the context: {context}, answer: {question}"

    try:
        # 🚀 UPDATE: Using the 2.5 series model which supports 1M+ context
        # Standard format in 2026 is just the ID string
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        # Fallback to 2.0 if 2.5 is not yet in your region
        if "404" in str(e):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt
                )
                return response.text.strip()
            except Exception as inner_e:
                return f"AI Error: {str(inner_e)}"
        return f"AI Error: {str(e)}"