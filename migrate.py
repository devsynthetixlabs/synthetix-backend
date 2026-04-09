import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Connections
# Ensure your .env has DATABASE_URL (Postgres)
PG_URL = os.getenv("DATABASE_URL")
# Point to your local file
SQLITE_URL = "sqlite:///database/synthetix.db" 

pg_engine = create_engine(PG_URL, connect_args={"sslmode": "require"})
sqlite_engine = create_engine(SQLITE_URL)

def migrate():
    # List the tables you want to move
    # Example: ['users', 'queries', 'stocks']
    tables = ['invoices', 'items', 'tax_logistics'] 

    for table in tables:
        print(f"📦 Migrating table: {table}...")
        try:
            # Use Pandas to read from SQLite and write to Postgres
            # This is the fastest way to migrate data types correctly
            df = pd.read_sql_table(table, sqlite_engine)
            
            # 'append' keeps existing data, 'replace' starts fresh
            df.to_sql(table, pg_engine, if_exists='append', index=False)
            print(f"✅ {table} migrated successfully.")
        except Exception as e:
            print(f"⚠️ Could not migrate {table}: {e}")

if __name__ == "__main__":
    migrate()