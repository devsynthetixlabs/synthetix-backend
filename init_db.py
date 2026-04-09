import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Use your standard .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def migrate():
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in .env")
        return

    # Create engine with SSL for Neon/Vercel
    engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

    # Using a context manager for the connection
    with engine.connect() as conn:
        print("Connected! Creating tables...")
        
        # 1. INVOICES
        # Note: Postgres uses NUMERIC or DECIMAL instead of REAL for precision
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_number TEXT,
                date TEXT,
                company_name TEXT,
                gst_no TEXT,
                invoice_amount DECIMAL,
                freight DECIMAL,
                packing_forwarding DECIMAL,
                packing_charges DECIMAL,
                round_off DECIMAL,
                igst DECIMAL,
                cgst DECIMAL,
                sgst DECIMAL,
                year TEXT,
                PRIMARY KEY (invoice_number, year)
            )
        '''))

        # 2. ITEMS
        # SERIAL is the Postgres way to do AUTOINCREMENT
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS items (
                item_id SERIAL PRIMARY KEY,
                invoice_number TEXT,
                year TEXT,
                description TEXT,
                quantity DECIMAL,
                rate DECIMAL,
                amount DECIMAL,
                discount DECIMAL,
                FOREIGN KEY (invoice_number, year) REFERENCES invoices (invoice_number, year) ON DELETE CASCADE
            )
        '''))

        # 3. TAX & LOGISTICS
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS tax_logistics (
                entry_id SERIAL PRIMARY KEY,
                invoice_number TEXT,
                year TEXT,
                type TEXT,
                amount DECIMAL,
                FOREIGN KEY (invoice_number, year) REFERENCES invoices (invoice_number, year) ON DELETE CASCADE
            )
        '''))
        
        conn.commit()
        print("🚀 Database tables created/updated successfully in Vercel!")

if __name__ == "__main__":
    migrate()