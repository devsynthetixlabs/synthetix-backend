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

# 1. Get the absolute path of the current file (sql_engine.py)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

db = SQLDatabase.from_uri(DATABASE_URL, engine_args={"connect_args": {"sslmode": "require"}})

# 3. Initialize the Brain
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0,
    model_kwargs={"response_format": {"type": "text"}} # Ensures it stays as text
)

# 4. Create the Question-to-SQL Chain
chain = create_sql_query_chain(llm, db)

sql_writer_prompt = ChatPromptTemplate.from_template("""
    You are an expert SQLite analyst for a sales database.

    SCHEMA:

    invoices(
    invoice_number, date, company_name, gst_no,
    invoice_amount, freight, packing_forwarding,
    packing_charges, round_off, igst, cgst, sgst, year
    )

    items(
    item_id, invoice_number, year,
    description, quantity, rate, amount, discount
    )

    RELATION:
    items.invoice_number = invoices.invoice_number
    AND items.year = invoices.year

    CORE RULES:
    - Generate ONLY one valid SELECT query
    - Return ONLY SQL (no explanation, no semicolon)
    - Use only columns from schema (no hallucination)
    - Use simplest query possible (avoid unnecessary JOINs/subqueries)
    - Treat each question independently (no previous context)
    - MATCHING: Never use = for company_name. ALWAYS use LIKE '%name%'.
    - CLEANING: If the user provides a short name (e.g., "A"), search for LIKE '%A%'.

    USAGE RULES:
    - Use invoices for invoice/customer/revenue queries
    - Use items for item/quantity/description queries
    - Use JOIN only when both invoice + item data are needed:
    items i JOIN invoices iv 
    ON i.invoice_number = iv.invoice_number AND i.year = iv.year
    - Use iv.company_name for customer (not items)

    AGGREGATION:
    - Use SUM, COUNT, AVG, MAX when needed
    - Use GROUP BY correctly when aggregating
    - "customers" = COUNT(DISTINCT company_name)
    - "revenue" = SUM(invoice_amount)
    - "freight" = SUM(freight)

    ORDERING & LIMIT:
    - "highest", "top" → ORDER BY DESC
    - "lowest", "bottom" → ORDER BY ASC
    - Always use LIMIT for ranked queries
    - Do NOT return unnecessary large datasets

    MULTI-METRIC:
    - If comparing metrics:
    use MAX() for single invoice
    use SUM() for total revenue
    - Return all required values in same query

    FINANCIAL YEAR (FY):
    - FY = April 1 → March 31
    - FY 2021-22 → date >= '2021-04-01' AND date <= '2022-03-31'
    - Use date column for FY filtering (not just year)

    CONSTRAINTS:
    - Do NOT reuse values from previous queries
    - Do NOT add filters not mentioned in question

    FOR QUESTIONS asking:
    - "largest", "highest", "top"

    ALWAYS SELECT:
    - both identifier (company_name)
    - AND numeric value (invoice_amount or SUM)

    Never return only name without value

    EXAMPLES:

    Q: Total revenue  
    A:
    SELECT SUM(invoice_amount) FROM invoices

    Q: Top 5 customers  
    A:
    SELECT company_name, SUM(invoice_amount) AS total
    FROM invoices
    GROUP BY company_name
    ORDER BY total DESC
    LIMIT 5

    Q: Most sold item  
    A:
    SELECT description, SUM(quantity) AS total_qty
    FROM items
    GROUP BY description
    ORDER BY total_qty DESC
    LIMIT 1

    Q: Largest order in 2021  
    A:
    SELECT company_name, invoice_amount
    FROM invoices
    WHERE year = '2021'
    ORDER BY invoice_amount DESC
    LIMIT 1

    Q: Total revenue in FY 2021-22  
    A:
    SELECT SUM(invoice_amount)
    FROM invoices
    WHERE date >= '2021-04-01' AND date <= '2022-03-31'

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

def ask_cfo(question):
    try:
        # 1. Generate SQL
        sql_writer_chain = sql_writer_prompt | llm
        raw_sql = sql_writer_chain.invoke({"question": question}).content
        sql_query = clean_sql(raw_sql)

        # 2. Execute SQL - Direct Engine call is safer than db.run()
        with db._engine.connect() as connection:
            db_data = connection.execute(text(sql_query)).fetchall()
        
        if not db_data:
            return "No data found for this period."

        # 3. THE "LOCK": Format numbers as strings in Python
        # This ensures the LLM sees "₹8,58,30,762.86" as text, not a number it can change
        processed_data = []
        for row in db_data:
            new_row = []
            for item in row:
                if isinstance(item, (int, float, Decimal)):
                    new_row.append(format_inr(item)) # Converts 85830762.86 -> "₹8,58,30,762.86"
                else:
                    new_row.append(str(item))
            processed_data.append(new_row)

        # 4. Narrate using YOUR specific prompt
        narrator_chain = narrator_prompt | llm
        answer = narrator_chain.invoke({
            "result": str(processed_data), # Sending the already-formatted strings
            "question": question
        })

        return {
            "answer": answer.content,
            "sql": sql_query
        }

    except Exception as e:
        return f"Error: {str(e)}"
