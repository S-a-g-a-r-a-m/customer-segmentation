import pandas as pd

DATA_PATH = "data/Retail_Transactions_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

rfm = df.groupby("Customer_Name").agg(
    LastPurchase=("Date", "max"),
    Frequency=("Transaction_ID", "count"),
    Monetary=("Total_Cost", "sum")
)
analysis_date = df["Date"].max() + pd.Timedelta(days=1)

rfm["Recency"] = (
    analysis_date - rfm["LastPurchase"]
).dt.days
print(rfm.head())
print("\nRFM summary:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())
print("\nUnique customers in RFM:")
print(rfm.index.nunique())

print("\nRFM shape:")
print(rfm.shape)