"""Practical 1: Identify association between independent and dependent variables."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    X, y = make_regression(n_samples=250, n_features=3, noise=18, random_state=42)
    df = pd.DataFrame(X, columns=["study_hours", "attendance", "practice_score"])
    df["exam_score"] = y
    print("Sample data:\n", df.head(), "\n")
    print("Correlation with dependent variable (exam_score):")
    print(df.corr(numeric_only=True)["exam_score"].sort_values(ascending=False))

    X_train, X_test, y_train, y_test = train_test_split(
        df[["study_hours", "attendance", "practice_score"]], df["exam_score"],
        test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nDependent variable: exam_score")
    print("Independent variables:", list(X_train.columns))
    print("Coefficients:", dict(zip(X_train.columns, model.coef_.round(3))))
    print(f"R² score: {r2_score(y_test, predictions):.3f}")
    df.plot.scatter(x="study_hours", y="exam_score", title="Study hours vs exam score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
