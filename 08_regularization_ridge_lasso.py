"""Practical 8: Use Ridge and Lasso regularization to reduce overfitting."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

def main():
    X, y = make_regression(n_samples=180, n_features=8, noise=25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
        "Ridge (alpha=10)": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=10)),
        "Lasso (alpha=1)": make_pipeline(StandardScaler(), PolynomialFeatures(2), Lasso(alpha=1, max_iter=20000)),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_rmse = mean_squared_error(y_train, model.predict(X_train)) ** 0.5
        test_rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
        print(f"{name:22s} | train RMSE: {train_rmse:8.3f} | test RMSE: {test_rmse:8.3f}")
    print("\nLower test RMSE is better; regularization strength may need tuning.")

if __name__ == "__main__":
    main()
