from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

def _fetch_sales_context(question: str, tenant_id: str) -> str:
    """Run a quick SQL query to get relevant sales data for context."""
    from engine.sql_engine import ask_cfo
    
    context_questions = [
        f"What are the top 5 clients by revenue for {tenant_id}?",
        f"What is the total revenue and year-over-year trend?",
    ]
    
    results = []
    for q in context_questions:
        try:
            res = ask_cfo(q, tenant_id)
            if isinstance(res, dict) and 'answer' in res:
                results.append(res['answer'])
        except Exception:
            pass
    
    return " ".join(results) if results else "No specific sales data available."

def ask_strategy(question: str, tenant_id: str) -> str:
    """
    Strategy engine: fetches sales data, then generates a strategy response.
    Uses a single LLM call instead of CrewAI multi-agent to save dependencies.
    """
    try:
        sales_context = _fetch_sales_context(question, tenant_id)
    except Exception:
        sales_context = "Sales data could not be retrieved."

    prompt = ChatPromptTemplate.from_template("""
        You are a senior business consultant analyzing sales data.

        SALES DATA CONTEXT:
        {sales_context}

        USER QUESTION:
        {question}

        RULES:
        - Base your answer on the sales data provided above
        - If data is unavailable, say so directly
        - Reference specific company names and ₹ amounts from the data
        - Keep the answer to 3-5 sentences max
        - Provide actionable, specific advice
        - Do NOT invent numbers or companies

        Answer:
    """)

    try:
        chain = prompt | llm
        response = chain.invoke({
            "sales_context": sales_context,
            "question": question,
        })
        return response.content
    except Exception as e:
        return f"Strategy analysis unavailable. Error: {str(e)}"
