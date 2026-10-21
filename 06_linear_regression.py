"""Practical 6: Implement Linear Regression."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=250, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Intercept:", round(model.intercept_, 3))
    print("Coefficient:", model.coef_.round(3))
    print(f"MAE: {mean_absolute_error(y_test, pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.3f}")
    print(f"R²: {r2_score(y_test, pred):.3f}")
    plt.scatter(X_test[:, 0], y_test, label="Actual")
    order = X_test[:, 0].argsort()
    plt.plot(X_test[order, 0], pred[order], label="Predicted line")
    plt.title("Linear Regression")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
