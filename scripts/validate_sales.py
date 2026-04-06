import pandas as pd
import re

def clean_val(val):
    if pd.isna(val) or val == "": return 0.0
    # Removes 'Amount', commas, etc.
    res = re.findall(r"[-+]?\d*\.\d+|\d+", str(val).replace(',', ''))
    return float(res[0]) if res else 0.0

def run_pre_flight_check(file_path):
    # Load the data (works for .xlsx or .csv)
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    
    report = {
        "total_rows": len(df),
        "passed": 0,
        "failed": 0,
        "errors": []
    }

    # Group by Invoice Number to handle multi-line items
    grouped = df.groupby('Invoice_Number')

    for inv_no, group in grouped:
        # 1. Header Data
        first_row = group.iloc[0]
        grand_total = clean_val(first_row.get('Invoice_Amount', 0))
        
        # 2. Sum of Items
        # Note: In your Tally export, sometimes 'Amount' is the line total
        items_total = group['Rate'].apply(clean_val).sum() * group['Quantity'].apply(clean_val).sum() 
        # Alternatively, if 'Amount' column exists per line:
        line_sum = group['Amount'].apply(clean_val).sum()

        # 3. Sum of Extras (Taxes & Logistics)
        tax_keys = ['IGST @ 18 %', 'CGST @ 9 %', 'SGST @ 9 %', 'Freight', 'Packing & Forwarding']
        extras_total = 0
        for key in tax_keys:
            if key in group.columns:
                extras_total += clean_val(group[key].iloc[0])

        # 4. Validation Math
        calculated_total = line_sum + extras_total
        diff = abs(grand_total - calculated_total)

        if diff < 2.0: # Allowing 2 Rupee margin for Round-Off
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["errors"].append({
                "inv": inv_no,
                "expected": grand_total,
                "calculated": calculated_total,
                "diff": diff
            })

    print(f"--- 🚀 PRE-FLIGHT RESULTS ---")
    print(f"Total Unique Invoices: {len(grouped)}")
    print(f"✅ Passed: {report['passed']}")
    print(f"❌ Failed: {report['failed']}")
    
    if report["failed"] > 0:
        print("\n--- 🚩 TOP ERRORS TO FIX ---")
        for err in report["errors"][:5]:
            print(f"Invoice {err['inv']}: Expected ₹{err['expected']}, but got ₹{err['calculated']} (Diff: ₹{err['diff']})")

# Run it on your file
run_pre_flight_check('../data/financials/Sales_Invoice_Registers_For_FY 2019-20.XLS')