import os
import locale
import re
import ast
from dotenv import load_dotenv
from thefuzz import process
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import text
from decimal import Decimal
from database.db_helper import get_db_schema
from engine.core import db

def format_inr(number):
    try:
        # Convert to float and round to 2 decimal places
        n = f"{float(number):.2f}"
        whole, fraction = n.split('.')
        
        # Indian Numbering Logic: Last 3 digits grouped, then groups of 2
        if len(whole) > 3:
            last_three = whole[-3:]
            remaining = whole[:-3]
            # Add commas every 2 digits for the remaining part
            remaining = re.sub(r'(?<=.)(?=(..)+$)', ',', remaining)
            whole = remaining + ',' + last_three
        
        return f"₹{whole}.{fraction}"
    except Exception:
        return f"₹{number}"

# TEST: format_inr(3305121) -> '₹33,05,121.00'

def find_real_company_name(short_name):
    query = text("SELECT DISTINCT company_name FROM invoices WHERE company_name ILIKE :name LIMIT 1")
    try:
        # Using the underlying engine for a clean result
        with db._engine.connect() as connection:
            result = connection.execute(query, {"name": f"%{short_name}%"}).fetchone()
            if result:
                return result[0] # Returns the string directly
    except Exception as e:
        print(f"Lookup error: {e}")
    return short_name

# TEST: find_real_company_name("PRALCKA") -> "PRALCKA MACHINERY MANUFACTURING PVT. LTD"

load_dotenv()

# 3. Initialize the Brain
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0,
    model_kwargs={"response_format": {"type": "text"}} # Ensures it stays as text
)

# 4. Create the Question-to-SQL Chain
chain = create_sql_query_chain(llm, db)

current_schema = get_db_schema(db._engine)

sql_writer_prompt = ChatPromptTemplate.from_template("""
    You are an expert PostgreSQL analyst for a sales database.
    (Note: You are using SQLAlchemy to interact with a Postgres database).

    DATABASE_SCHEMA:
    {schema}

    TENANT_CONTEXT:
    Current Tenant ID: {tenant_id}

    RELATION:
    items.invoice_number = invoices.invoice_number
    AND items.year = invoices.year

    POSTGRES TYPE RULES:
    - COLUMN 'year': This is a VARCHAR/TEXT column. 
      * For filtering: Use single quotes: WHERE year = '2023'
      * For math (YoY): You MUST cast it: CAST(year AS INTEGER) - 1
    - ILIKE: Use ILIKE for case-insensitive text matching.
    - DATE: Use standard ISO format 'YYYY-MM-DD'.

    CORE RULES:
    - Generate ONLY one valid SELECT query.
    - Return ONLY SQL (no explanation, no semicolon).
    - Use only columns from the provided schema.
    - MATCHING: Never use = for company_name. ALWAYS use ILIKE '%name%'.

    USAGE RULES:
    - Use 'invoices' for revenue, customer, or shipping queries.
    - Use 'items' for quantity, product description, or rate queries.
    - Use JOIN only when data from both tables is required.

    AGGREGATION & MATH:
    - "revenue" = SUM(invoice_amount)
    - "customers" = COUNT(DISTINCT company_name)
    - Handle Year-over-Year (YoY) by joining the table to itself or using subqueries with explicit CAST(year AS INTEGER).

    EXAMPLES:

    Q: Total revenue  
    A: SELECT SUM(invoice_amount) FROM invoices

    Q: Largest order in 2021  
    A: SELECT company_name, invoice_amount FROM invoices WHERE year = '2021' ORDER BY invoice_amount DESC LIMIT 1

    Q: Which company sales dropped year-on-year?
    A: SELECT a.company_name, SUM(a.invoice_amount) as current, SUM(b.invoice_amount) as previous 
       FROM invoices a 
       JOIN invoices b ON a.company_name = b.company_name 
       WHERE a.year = '2023' AND b.year = '2022' 
       GROUP BY a.company_name 
       HAVING SUM(a.invoice_amount) < SUM(b.invoice_amount)

    Q: Sales for company X in FY 2021-22
    A: SELECT SUM(invoice_amount) FROM invoices WHERE company_name ILIKE '%X%' AND date >= '2021-04-01' AND date <= '2022-03-31'

    User Question:
    {question}

    SQL:
""")

narrator_prompt = ChatPromptTemplate.from_template("""
    You are a business/data analyst providing clear and direct answers.

    User Question: {question}
    SQL Result: {result}

    RULES:
    - Use ONLY the SQL result provided
    - If result is empty/null, say "No data available"
    - Do NOT invent data, tables, or assumptions
    - Do NOT modify, round, or create numbers
    - Do NOT analyze beyond the given rows
    - Start directly with the answer (no introductions like "Based on..." or "According to...")
    - You MUST use the EXACT currency string provided in the data.
    - Do NOT change ₹33,05,121.00 to any other number.
    - If you change the digits or the commas, the report is WRONG.
    - Provide a single, direct sentence.

    CURRENCY:
    - Use INR (₹) only
    - Format in Indian style (₹1,23,45,678)
    - Never use $

    EXAMPLES:
    Input: [["ABC Ltd", 1000000]]
    Output: "ABC Ltd generated ₹10,00,000 in revenue."

    - "The total revenue is ₹45,00,000."
    - "PACE PACKAGING MACHINES PVT LTD generated the highest revenue at ₹76,29,880."
    - "There are 1,358 invoices in total."

    CRITICAL RULE:
    - If numeric value is NOT present in SQL result, say:
    "Amount data not available"
    - NEVER generate or guess numbers

    Answer:
""")

def clean_sql(raw_sql: str):
    # 1. Remove Markdown code blocks
    sql = re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()
    
    # 2. 🚨 CRITICAL: Extract only the part starting with SELECT
    # This ignores any "Here is the query:" chatter from the LLM
    match = re.search(r"(SELECT[\s\S]+)", sql, re.IGNORECASE)
    if match:
        sql = match.group(1)

    # 3. Remove trailing symbols
    sql = sql.split(';')[0].strip()
    return sql

def validate_sql(query: str):
    query_upper = query.upper()

    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]

    if any(word in query_upper for word in forbidden):
        raise ValueError("Unsafe query detected")

    # 🚨 Prevent multiple statements
    if ";" in query.strip()[:-1]:
        raise ValueError("Multiple SQL statements not allowed")

    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT allowed")

def enforce_limit(query: str):
    query = query.rstrip(";")  # extra safety

    if "LIMIT" not in query.upper():
        query += " LIMIT 100"

    return query

def normalize_result(result):
    if isinstance(result, str):
        result = eval(result)

    # If single row
    if isinstance(result, list) and len(result) == 1:
        return result[0]   # ✅ return full row

    return result

def execute_query(query):
    result = db.run(query)

    if isinstance(result, str):
        result = eval(result)

    return result

def ask_cfo(question, tenant_id):
    max_retries = 3
    error_log = ""
    last_sql = ""
    
    for attempt in range(max_retries):
        try:
            # Step 1: Generate SQL (feeding back errors if they exist)
            sql_writer_chain = sql_writer_prompt | llm
            raw_sql = sql_writer_chain.invoke({
                "question": f"{question} {error_log}", 
                "schema": current_schema,
                "tenant_id": tenant_id,
            }).content
            
            sql_query = clean_sql(raw_sql)
            last_sql = sql_query # Store for debugging
            validate_sql(sql_query)

            # Step 2: Execute against Neon
            with db._engine.connect() as connection:
                db_data = connection.execute(text(sql_query)).fetchall()
            
            # Step 3: SUCCESS - Process and return
            formatted_answer = process_and_narrate(db_data, question)
            return {
                "answer": formatted_answer,
                "sql": last_sql
            }

        except Exception as e:
            # Step 4: FAILURE - Log error and loop back
            error_log = f"\n(Previous Attempt Error: {str(e)})"
            print(f"🔄 Retrying SQL (Attempt {attempt + 1}/3)...")
            
    return {
        "answer": "I attempted to query the data but ran into a structural issue I couldn't resolve.",
        "sql": last_sql
    }

def process_and_narrate(db_data, question):
    """
    Processes raw SQL rows into formatted strings and calls the LLM Narrator.
    """
    if not db_data:
        return "No data found for this period."

    # 1. Format numbers using your format_inr logic
    processed_data = []
    for row in db_data:
        new_row = []
        for item in row:
            if isinstance(item, (int, float, Decimal)):
                new_row.append(format_inr(item)) # Uses your existing ₹ logic
            else:
                new_row.append(str(item))
        processed_data.append(new_row)

    # 2. Call the Narrator Chain
    narrator_chain = narrator_prompt | llm
    answer = narrator_chain.invoke({
        "result": str(processed_data),
        "question": question
    })

    return answer.content
