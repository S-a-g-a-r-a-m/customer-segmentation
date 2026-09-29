import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


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
# 3. Select ML features
# =========================

X = rfm[[
    "Recency",
    "Frequency",
    "Monetary"
]]


# =========================
# 4. Scale features
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# 5. Display original/scaled data
# =========================

print("Original features:")
print(X.head())

print("\nScaled features:")
print(X_scaled[:5])


# =========================
# 6. Elbow Method
# =========================

inertias = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)

    print(
        f"K={k}, Inertia={kmeans.inertia_:.2f}"
    )


# =========================
# 7. Plot Elbow Curve
# =========================

k_values = range(2, 9)

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()


# =========================
# 8. Compare K values
#    using Silhouette Score
# =========================

print("\nSilhouette Score Comparison:")

for k in range(3, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=10000,
        random_state=42
    )

    print(f"\nK={k}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {score:.4f}")


# =========================
# 9. Compare cluster profiles
# =========================

for k in range(3, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(X_scaled)

    # Copy RFM data so we can attach cluster labels
    profile = X.copy()

    profile["Cluster"] = labels

    print(f"\n{'=' * 50}")
    print(f"Cluster Profiles for K={k}")
    print(f"{'=' * 50}")

    cluster_profile = profile.groupby("Cluster").agg(
        Customers=("Recency", "size"),

        Avg_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),

        Avg_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),

        Avg_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median")
    )

    print(cluster_profile.round(2))


# =========================
# 10. Final K-Means
# =========================

# Based on:
# - Elbow method
# - Silhouette score
# - Cluster profile interpretation
#
# We use K=4 as the final working model.

FINAL_K = 4

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(X_scaled)


# =========================
# 11. Attach final clusters
# =========================

rfm["Cluster"] = labels


# =========================
# 12. Final cluster counts
# =========================

print("\nFinal Cluster Counts:")

cluster_counts = (
    rfm["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


# =========================
# 13. Final cluster profile
# =========================

print("\nFinal Cluster Profile:")

final_profile = rfm.groupby("Cluster").agg(
    Customers=("Recency", "size"),

    Avg_Recency=("Recency", "mean"),
    Median_Recency=("Recency", "median"),

    Avg_Frequency=("Frequency", "mean"),
    Median_Frequency=("Frequency", "median"),

    Avg_Monetary=("Monetary", "mean"),
    Median_Monetary=("Monetary", "median")
)

print(final_profile.round(2))


# =========================
# 14. RFM percentiles
# =========================

print("\nRFM Percentiles:")

print(
    X.quantile(
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)

# =========================
# 15. PCA Visualization
# =========================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum())


# =========================
# 16. Sample customers
#    for visualization
# =========================

sample_size = 20000

sample_indices = (
    pd.Series(range(len(X_pca)))
    .sample(
        n=sample_size,
        random_state=42
    )
    .values
)

X_pca_sample = X_pca[sample_indices]
labels_sample = labels[sample_indices]


# =========================
# 17. PCA plot
# =========================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca_sample[:, 0],
    X_pca_sample[:, 1],
    c=labels_sample,
    s=8,
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Customer Segments - PCA Visualization"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.show()