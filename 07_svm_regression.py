"""Practical 7: Implement Support Vector Regression (SVR)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    rng = np.random.default_rng(42)
    X = np.sort(5 * rng.random((250, 1)), axis=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.15, X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.3f}")
    print(f"R²: {r2_score(y_test, pred):.3f}")
    grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    plt.scatter(X[:, 0], y, s=15, alpha=0.6, label="Data")
    plt.plot(grid[:, 0], model.predict(grid), label="SVR prediction")
    plt.title("Support Vector Regression (RBF kernel)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
