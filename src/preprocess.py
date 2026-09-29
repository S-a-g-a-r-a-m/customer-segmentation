import pandas as pd

DATA_PATH = "data/Retail_Transactions_Dataset.csv"

df = pd.read_csv(DATA_PATH)

# Convert Date from string to datetime
df["Date"] = pd.to_datetime(df["Date"])

print(df.dtypes)
print("\nDate range:")
print(df["Date"].min())
print(df["Date"].max())