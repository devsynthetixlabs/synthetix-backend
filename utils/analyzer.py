import pandas as pd

class FinancialAnalyzer:
    def __init__(self, df, amt_col):
        self.df = df.dropna(subset=[amt_col]).copy()
        self.amt_col = amt_col
        # Ensure Date is datetime for time-series analysis
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df['year'] = self.df['Date'].dt.year.astype(str)

    # TOOL 1: Growth & Variance (The "Why" Tool)
    def get_growth_metrics(self):
        summary = self.df.groupby('year')[self.amt_col].sum()
        growth = summary.pct_change() * 100
        return growth.dropna().to_dict()

    # TOOL 2: Customer Concentration (The "Risk" Tool)
    def get_top_customers(self, year=None):
        target_df = self.df[self.df['year'] == year] if year else self.df
        # Assuming 'Particulars' contains customer names
        return target_df.groupby('Particulars')[self.amt_col].sum().nlargest(5).to_dict()

    # TOOL 3: Monthly Seasonality (The "Timing" Tool)
    def get_monthly_trends(self, year):
        year_df = self.df[self.df['year'] == year]
        return year_df.groupby(year_df['Date'].dt.month_name())[self.amt_col].sum().to_dict()