import pandas as pd
import re

def is_date(val):
    try:
        pd.to_datetime(val)
        return True
    except:
        return False

def clean_tally_excel(file_path):
    df = pd.read_excel(file_path, header=None, skiprows=6)

    current_company = None
    current_invoice = {}

    cleaned_rows = []

    for _, row in df.iterrows():
        row = row.tolist()
        first_cell = str(row[0]).strip()

        # 🏢 Detect company
        if first_cell.startswith("*") and "Total" not in first_cell:
            match = re.search(r'-(.*)', first_cell)
            if match:
                current_company = match.group(1).strip()
            continue

        # ❌ Skip totals
        if "Total of Party" in first_cell:
            continue

        # ❌ Skip header row
        if "Item Description" in str(row):
            continue

        # 📄 Detect NEW invoice row (robust)
        if is_date(row[0]) and pd.notna(row[1]):
            current_invoice = {
                "Date": row[0],
                "Invoice": row[1],
                "GST": row[2],
                "Invoice_Amount": row[3]
            }

            item = row[4]
            qty = row[5]
            rate = row[6]

        # 📦 Continuation row
        elif pd.notna(row[4]) and current_invoice:
            item = row[4]
            qty = row[5]
            rate = row[6]

        else:
            continue

        cleaned_rows.append({
            "Company": current_company,
            "Date": current_invoice.get("Date"),
            "Invoice_Number": current_invoice.get("Invoice"),
            "GST": current_invoice.get("GST"),
            "Invoice_Amount": current_invoice.get("Invoice_Amount"),
            "Item": item,
            "Quantity": qty,
            "Rate": rate
        })

    df_clean = pd.DataFrame(cleaned_rows)

    # 🧹 Convert numeric columns safely
    df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce")
    df_clean["Rate"] = pd.to_numeric(df_clean["Rate"], errors="coerce")

    return df_clean

def validate_data(df):
    errors = []

    if df["Company"].isna().sum() > 0:
        errors.append("Missing company names")

    if df["Invoice_Number"].isna().sum() > 0:
        errors.append("Missing invoice numbers")

    if df["Quantity"].isna().sum() > 0:
        errors.append("Invalid quantity values")

    if df["Rate"].isna().sum() > 0:
        errors.append("Invalid rate values")

    return errors