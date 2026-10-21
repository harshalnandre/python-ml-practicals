"""Practical 3: Reduce dimensionality using Principal Component Analysis (PCA)."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    print("Original dimensions:", X.shape[1])
    print("Reduced dimensions:", reduced.shape[1])
    print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    plt.scatter(reduced[:, 0], reduced[:, 1], c=data.target, s=35)
    plt.title("Wine dataset projected onto first two principal components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
