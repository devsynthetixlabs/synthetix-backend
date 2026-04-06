import os
import locale
import re
from dotenv import load_dotenv
from thefuzz import process
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate

# Set the locale to Indian English
try:
    locale.setlocale(locale.LC_ALL, 'en_IN')
except:
    # Fallback for systems where en_IN isn't installed
    locale.setlocale(locale.LC_ALL, '') 

def format_inr(n):
    try:
        # 'n' is the float from your SQL result
        return locale.currency(float(n), symbol=True, grouping=True)
    except:
        return f"₹{n}"

# TEST: format_inr(3305121) -> '₹33,05,121.00'

def find_real_company_name(short_name):
    # Search the DB for names containing the user's input
    query = f"SELECT DISTINCT company_name FROM invoices WHERE company_name LIKE '%{short_name}%' LIMIT 1"
    res = db.run(query)
    # db.run usually returns a string representation of a list of tuples like "[('FULL NAME',)]"
    if res and "[]" not in res:
        # Extract the name from the string result
        import ast
        try:
            actual_name = ast.literal_eval(res)[0][0]
            return actual_name
        except:
            return short_name
    return short_name

# TEST: find_real_company_name("PRALCKA") -> "PRALCKA MACHINERY MANUFACTURING PVT. LTD"

load_dotenv()
# 1. Get the absolute path of the current file (sql_engine.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the database in the sibling 'data' folder
db_path = os.path.abspath(os.path.join(current_dir, "..", "data", "synthetix.db"))

# 3. Connect using the absolute path
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

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
    # Look for capitalized words that might be companies
    words = re.findall(r'\b[A-Z]{2,}\b', question)
    
    refined_question = question
    for word in words:
        # Check if this word is a nickname for a real company
        real_name = find_real_company_name(word)
        if real_name != word:
            refined_question = refined_question.replace(word, real_name)
    try:
        # 1. Generate SQL
        sql_writer_chain = sql_writer_prompt | llm
        raw_sql = sql_writer_chain.invoke({"question": refined_question}).content
        sql_query = clean_sql(raw_sql)
        validate_sql(sql_query)
        
        # 2. Execute SQL
        db_data = execute_query(sql_query)
        if not db_data:
            return "No data found for this period."

        # 3. FIX: PRE-FORMAT THE DATA (The "Anti-Hallucination" Step)
        # We process the list/result in Python to handle the currency formatting
        processed_data = []
        if isinstance(db_data, list):
            for row in db_data:
                new_row = []
                for item in row:
                    # If it's a large number, format it as INR immediately
                    if isinstance(item, (int, float)) and item > 100:
                        new_row.append(format_inr(item))
                    else:
                        new_row.append(item)
                processed_data.append(new_row)
        else:
            processed_data = db_data

        # 4. Narrate the already-formatted data
        narrator_chain = narrator_prompt | llm
        answer = narrator_chain.invoke({
            "result": processed_data, # Send the PRE-FORMATTED data
            "question": refined_question
        })

        return {
            "sql": sql_query,
            "raw_result": db_data,
            "processed_result": processed_data,
            "answer": answer.content
        }

    except Exception as e:
        return f"Error: {str(e)}"