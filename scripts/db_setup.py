import sqlite3
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "synthetix.db")

def setup_synthetix_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Start Fresh
    cursor.execute("DROP TABLE IF EXISTS items")
    cursor.execute("DROP TABLE IF EXISTS tax_logistics")
    cursor.execute("DROP TABLE IF EXISTS invoices")
    cursor.execute("DROP TABLE IF EXISTS invoice_items")

    # 1. INVOICES (Matches your 'invoices_table' list)
    cursor.execute('''
        CREATE TABLE invoices (
            invoice_number TEXT,
            date TEXT,
            company_name TEXT,
            gst_no TEXT,
            invoice_amount REAL,
            freight REAL,
            packing_forwarding REAL,
            packing_charges REAL,
            round_off REAL,
            igst REAL,
            cgst REAL,
            sgst REAL,
            year TEXT,
            PRIMARY KEY (invoice_number, year) -- THIS IS THE FIX
        )
    ''')

    # 2. ITEMS (Matches your 'items_table' list)
    cursor.execute('''
        CREATE TABLE items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number TEXT,
            year TEXT,
            description TEXT,
            quantity REAL,
            rate REAL,
            amount REAL,
            discount REAL,
            FOREIGN KEY (invoice_number, year) REFERENCES invoices (invoice_number, year)
        )
    ''')

    # 3. TAX & LOGISTICS (Matches your 'tax_logistics_table' list)
    cursor.execute('''
        CREATE TABLE tax_logistics (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number TEXT,
            type TEXT,
            amount REAL,
            FOREIGN KEY (invoice_number) REFERENCES invoices (invoice_number) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Foundation matches Python logic at: {DB_PATH}")

if __name__ == "__main__":
    setup_synthetix_db()