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

        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS document_knowledge (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(255),
            file_name VARCHAR(255),
            content TEXT,           -- The full markdown/structured text
            category VARCHAR(50),   -- e.g., 'HR', 'Finance', 'Operations'
            metadata JSONB,         -- Stores page counts, upload date, etc.
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
        ))

        conn.execute(text("""
            ALTER TABLE document_knowledge 
            ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) DEFAULT 'synthetix_admin_internal';
        """))
        
        # 2. Update items
        conn.execute(text("""
            ALTER TABLE items 
            ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) DEFAULT 'synthetix_admin_internal';
        """))
        
        # 3. Update invoices
        conn.execute(text("""
            ALTER TABLE invoices 
            ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) DEFAULT 'synthetix_admin_internal';
        """))
        
        # 4. Create Index (Note: INDEXes don't support IF NOT EXISTS in all Postgres versions, 
        # so we wrap this carefully or check if it exists)
        try:
            conn.execute(text("CREATE INDEX idx_invoices_tenant ON invoices(tenant_id);"))
            conn.execute(text("CREATE INDEX idx_items_tenant ON items(tenant_id);"))
            conn.execute(text("CREATE INDEX idx_docs_tenant ON document_knowledge(tenant_id);"))
        except Exception as e:
            print(f"Index might already exist: {e}")
        
        conn.commit()
        print("🚀 Database tables created/updated successfully in Vercel!")

if __name__ == "__main__":
    migrate()