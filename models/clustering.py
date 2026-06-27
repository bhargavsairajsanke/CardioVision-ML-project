import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from preprocessing import preprocess_data

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Load Processed Data
# ==========================================================

X_train, X_test, y_train, y_test, scaler, df = preprocess_data()

# ==========================================================
# PCA (Reduce dimensions to 2D for visualization)
# ==========================================================

pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_train)

# ==========================================================
# KMeans Clustering
# ==========================================================

print("=" * 60)
print("Training KMeans...")
print("=" * 60)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_train)

# Save Model

with open(os.path.join(MODEL_DIR, "kmeans.pkl"), "wb") as file:
    pickle.dump(kmeans, file)

# Plot

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels
)

plt.title("KMeans Clusters")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "kmeans_clusters.png"
    )
)

plt.close()

# ==========================================================
# DBSCAN
# ==========================================================

print("=" * 60)
print("Training DBSCAN...")
print("=" * 60)

dbscan = DBSCAN(
    eps=1.5,
    min_samples=8
)

dbscan_labels = dbscan.fit_predict(X_train)

# Save Model

with open(os.path.join(MODEL_DIR, "dbscan.pkl"), "wb") as file:
    pickle.dump(dbscan, file)

# Plot

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=dbscan_labels
)

plt.title("DBSCAN Clusters")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGE_DIR,
        "dbscan_clusters.png"
    )
)

plt.close()

# ==========================================================
# Cluster Information
# ==========================================================

print("\nKMeans Cluster Counts")
print(pd.Series(kmeans_labels).value_counts())

print("\nDBSCAN Cluster Counts")
print(pd.Series(dbscan_labels).value_counts())

print("\nModels Saved Successfully!")

print("Graphs Saved Successfully!")

print("=" * 60)
print("Clustering Completed Successfully")
print("=" * 60)