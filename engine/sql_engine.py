import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
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

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0,
    model_kwargs={"response_format": {"type": "text"}}
)

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

    # Prevent multiple statements (check for semicolons outside string literals)
    semicolon_count = query.count(";")
    if semicolon_count > 1:
        raise ValueError("Multiple SQL statements not allowed")
    if semicolon_count == 1 and not query.strip().endswith(";"):
        raise ValueError("Multiple SQL statements not allowed")

    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT allowed")

def enforce_limit(query: str):
    query = query.rstrip(";")  # extra safety

    if "LIMIT" not in query.upper():
        query += " LIMIT 100"

    return query

def ask_cfo(question, tenant_id):
    max_retries = 3
    error_log = ""
    last_sql = ""
    
    # Fetch fresh schema each time (High severity fix)
    fresh_schema = get_db_schema(db._engine)
    
    for attempt in range(max_retries):
        try:
            # Step 1: Generate SQL (feeding back errors if they exist)
            sql_writer_chain = sql_writer_prompt | llm
            raw_sql = sql_writer_chain.invoke({
                "question": f"{question} {error_log}", 
                "schema": fresh_schema,
                "tenant_id": tenant_id,
            }).content
            
            sql_query = clean_sql(raw_sql)
            last_sql = sql_query # Store for debugging
            validate_sql(sql_query)

            # CRITICAL FIX: Enforce tenant isolation at SQL level
            # Inject tenant_id filter to prevent cross-tenant data leaks
            if "WHERE" in sql_query.upper():
                # Add tenant_id to existing WHERE clause
                sql_query = re.sub(
                    r"(WHERE\s+)",
                    f"\\1tenant_id = '{tenant_id}' AND ",
                    sql_query,
                    flags=re.IGNORECASE
                )
            else:
                # Add WHERE clause if none exists
                sql_query = sql_query.rstrip(";") + f" WHERE tenant_id = '{tenant_id}'"

            # Enforce limit
            sql_query = enforce_limit(sql_query)

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
