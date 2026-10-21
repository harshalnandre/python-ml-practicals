"""Practical 5: Apply K-Nearest Neighbours (KNN) classification."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Dataset: Iris (built into scikit-learn)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()
