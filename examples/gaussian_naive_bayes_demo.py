from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from forge.naive_bayes import GaussianNB


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nPredicted classes: {preds}")
    print(f"\nClass probabilities (first 5 samples):\n{probs[:5].round(4)}")
    print(f"\nLearned priors: {model.priors_.round(4)}")
    print(f"Learned means (per class, per feature):\n{model.means_.round(4)}")


if __name__ == "__main__":
    main()
