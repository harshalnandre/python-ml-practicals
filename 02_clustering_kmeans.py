"""Practical 2: Apply unsupervised clustering using K-Means."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def main():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.25, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    print("Cluster counts:", {int(i): int((labels == i).sum()) for i in set(labels)})
    print(f"Silhouette score: {silhouette_score(X_scaled, labels):.3f}")
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=35)
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
