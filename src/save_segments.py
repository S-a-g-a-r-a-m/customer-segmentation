import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# 1. Load dataset
# =========================

DATA_PATH = "data/Retail_Transactions_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])


# =========================
# 2. Create RFM features
# =========================

rfm = df.groupby("Customer_Name").agg(
    LastPurchase=("Date", "max"),
    Frequency=("Transaction_ID", "count"),
    Monetary=("Total_Cost", "sum")
)

analysis_date = df["Date"].max() + pd.Timedelta(days=1)

rfm["Recency"] = (
    analysis_date - rfm["LastPurchase"]
).dt.days


# =========================
# 3. Select features
# =========================

X = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]


# =========================
# 4. Scale features
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# 5. Final K-Means model
# =========================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init="auto"
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)


# =========================
# 6. Reset index
# =========================

rfm = rfm.reset_index()


# =========================
# 7. Save results
# =========================

OUTPUT_PATH = "data/customer_segments.csv"

rfm.to_csv(
    OUTPUT_PATH,
    index=False
)


print("Customer segmentation completed.")

print(f"\nSaved to: {OUTPUT_PATH}")

print(f"\nNumber of customers: {len(rfm)}")

print("\nCluster counts:")

print(
    rfm["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nFirst 5 customers:")

print(rfm.head())