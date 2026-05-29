import numpy as np
from forge.linear import LogisticRegression

def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)
    acc = np.mean(preds == y)

    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()