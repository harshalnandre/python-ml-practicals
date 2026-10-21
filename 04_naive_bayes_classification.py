"""Practical 4: Apply Gaussian Naive Bayes classification."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = make_pipeline(StandardScaler(), GaussianNB())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Dataset: Breast Cancer Wisconsin (built into scikit-learn)")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()
