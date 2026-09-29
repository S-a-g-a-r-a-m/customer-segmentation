import pandas as pd

df = pd.read_csv("data/Retail_Transactions_Dataset.csv")
print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types and non-null values:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())
print("\nUnique customers:")
print(df["Customer_Name"].nunique())

print("\nUnique transactions:")
print(df["Transaction_ID"].nunique())

print("\nTransactions per customer:")
print(df["Customer_Name"].value_counts().describe())
print("\nDate range:")
print(df["Date"].min())
print(df["Date"].max())

print("\nNumerical summary:")
print(df[["Total_Items", "Total_Cost"]].describe())