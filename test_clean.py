import pandas as pd
import re
from dateutil.parser import parse

def is_date(val):
    # Simple check to see if the cell looks like a date (e.g., starts with '20')
    return str(val).startswith('20')

def clean_num(val):
    """Removes commas and converts to float for SQLite REAL compatibility."""
    if pd.isna(val) or val == "" or val == "-":
        return 0.0
    try:
        # Remove commas and convert to float
        return float(str(val).replace(',', ''))
    except ValueError:
        return 0.0

def process_to_relational_tables(file_path):
    # Skipping first 6 rows as per your current logic
    df = pd.read_excel(file_path, header=None, skiprows=6)

    invoices_table = []
    items_table = []
    tax_logistics_table = []

    current_company = None
    current_year = None # Store the year here to use for items
    current_inv_no = None
    processed_invoices = set()

    for _, row in df.iterrows():
        row = row.tolist()
        first_cell = str(row[0]).strip()

        # 1. Detect Company
        if first_cell.startswith("*") and "Total" not in first_cell:
            match = re.search(r'-(.*)', first_cell)
            current_company = match.group(1).strip() if match else current_company
            continue

        # 2. Detect New Invoice Header
        if is_date(row[0]) and pd.notna(row[1]):
            current_inv_no = str(row[1]).strip()
            invoice_date = str(row[0])
            current_year = invoice_date[:4] # YEAR IS SET HERE
            
            if current_inv_no not in processed_invoices:

                invoices_table.append({
                    "invoice_number": current_inv_no,
                    "date": invoice_date,
                    "company_name": current_company, # Matches DB Column
                    "gst_no": str(row[2]),
                    "invoice_amount": clean_num(row[3]),
                    "freight": clean_num(row[10] or row[13]),
                    "packing_forwarding": clean_num(row[11]),
                    "packing_charges": clean_num(row[12]),
                    "round_off": clean_num(row[21]),
                    "igst": clean_num(row[17]),
                    "cgst": clean_num(row[19]),
                    "sgst": clean_num(row[20]),
                    "year": current_year # Added for AI performance
                })
                processed_invoices.add(current_inv_no)

                # 3. Extract Taxes/Logistics (Optional: helpful for categorical analysis)
                tax_map = {
                    "IGST": clean_num(row[17]),
                    "CGST": clean_num(row[19]),
                    "SGST": clean_num(row[20]),
                    "Freight": clean_num(row[11]),
                    "Forwarding": clean_num(row[12])
                }
                for tax_type, amt in tax_map.items():
                    if amt != 0:
                        tax_logistics_table.append({
                            "invoice_number": current_inv_no,
                            "type": tax_type,
                            "amount": amt
                        })

        # 4. Extract Item Lines
        if pd.notna(row[4]): # If Item Description exists
            items_table.append({
                "invoice_number": str(row[1]) if pd.notna(row[1]) else current_inv_no,
                "description": row[4], # Renamed from 'item_description' to match DB
                "quantity": clean_num(row[5]),
                "rate": clean_num(row[6]),
                "amount": clean_num(row[7]),
                "discount": clean_num(row[9]),
                "year": current_year
            })
    return pd.DataFrame(invoices_table), pd.DataFrame(items_table), pd.DataFrame(tax_logistics_table)

# Execute and Test
if __name__ == "__main__":
    file_path = './data/financials/Sales_Invoice_Registers_For_FY 2019-20.XLS'
    
    # 1. Run the processing
    df_inv, df_items, df_tax = process_to_relational_tables(file_path)

    # 2. FORCE OUTPUT TO TERMINAL
    print("\n" + "="*30)
    print("📊 EXTRACTION SUMMARY")
    print("="*30)
    print(f"Total Invoices Found: {len(df_inv)}")
    print(f"Total Line Items Found: {len(df_items)}")
    print(f"Total Tax Entries Found: {len(df_tax)}")
    print("="*30)

    if not df_inv.empty:
        print("\n👀 PREVIEW: FIRST 3 INVOICES")
        print(df_inv[['invoice_number', 'invoice_amount', 'date', 'company', 'gst_no']].head(3))
        
        print("\n📦 PREVIEW: FIRST 5 ITEMS")
        print(df_items[['invoice_number', 'item_description', 'rate','quantity', 'amount']].head(5))
    else:
        print("\n❌ No data was extracted. Check if 'skiprows=6' matches your file structure.")