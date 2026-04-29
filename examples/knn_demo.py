import numpy as np
from forge.neighbors import KNNClassifier


def main():
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = KNNClassifier(k=3)
    model.fit(X, y)

    preds = model.predict(X)
    acc = np.mean(preds == y)

    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()