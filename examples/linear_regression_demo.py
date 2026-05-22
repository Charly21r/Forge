import numpy as np
from forge.linear import LinearRegression


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    true_coef = np.array([2.0, -3.0])
    true_intercept = 1.5
    y = X @ true_coef + true_intercept + 0.1 * rng.standard_normal(100)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    preds = model.predict(X)
    mse = float(np.mean((preds - y) ** 2))

    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
