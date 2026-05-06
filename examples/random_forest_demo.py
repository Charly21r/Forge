from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from forge.ensemble import RandomForestClassifier


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_features="sqrt",
        max_depth=8,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"\nPredicted classes (first 10): {preds[:10]}")
    print(f"\nClass probabilities (first 5 samples):\n{probs[:5].round(4)}")

    importances = model.feature_importances_
    top_k = 5
    top_idx = importances.argsort()[::-1][:top_k]
    print(f"\nTop {top_k} most important features:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank}. {data.feature_names[idx]:<30s} ({importances[idx]:.4f})")


if __name__ == "__main__":
    main()
