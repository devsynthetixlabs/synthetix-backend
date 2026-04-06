import sqlite3
import pandas as pd
import glob
import os
import shutil
from test_clean import process_to_relational_tables

def batch_process_financials(folder_path, db_path='./data/synthetix.db'):
    processed_dir = os.path.join(folder_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(folder_path, "*.XLS*"))
    conn = sqlite3.connect(db_path)

    for file in all_files:
        try:
            inv_df, items_df, tax_df = process_to_relational_tables(file)
            
            # 1. Internal Cleanup (Current File)
            inv_df = inv_df.drop_duplicates(subset=['invoice_number', 'year'], keep='first')

            # 2. SAFE PUSH: Only add invoices that don't exist yet
            # We use a temp table to avoid the 'UNIQUE' crash
            inv_df.to_sql('temp_invoices', conn, if_exists='replace', index=False)
            
            conn.execute("""
                INSERT OR IGNORE INTO invoices 
                (invoice_number, year, date, company_name, gst_no, invoice_amount, freight, packing_forwarding, packing_charges, round_off, igst, cgst, sgst)
                SELECT invoice_number, year, date, company_name, gst_no, invoice_amount, freight, packing_forwarding, packing_charges, round_off, igst, cgst, sgst 
                FROM temp_invoices
            """)
            
            # 3. Push Items (Items don't have a unique constraint, so they just append)
            items_df.to_sql('items', conn, if_exists='append', index=False)
            tax_df.to_sql('tax_logistics', conn, if_exists='append', index=False)
            
            conn.execute("DROP TABLE temp_invoices")
            conn.commit()

            print(f"✅ Processed: {os.path.basename(file)}")
            shutil.move(file, os.path.join(processed_dir, os.path.basename(file)))
            
        except Exception as e:
            print(f"❌ Error with {file}: {e}")
    
    conn.close()

if __name__ == "__main__":
    data_dir = './data/financials/'
    batch_process_financials(data_dir)