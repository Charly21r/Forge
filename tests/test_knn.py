import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from forge.neighbors import KNNClassifier


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def simple_dataset():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])
    return X, y


@pytest.fixture
def random_dataset():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    return X, y


# --------------------------
# Basic functionality
# --------------------------
def test_knn_fit_predict_simple(simple_dataset):
    X, y = simple_dataset
    model = KNNClassifier(k=1).fit(X, y)
    preds = model.predict(np.array([[1.1]]))
    assert preds[0] == 1


def test_knn_score_matches_predictions(simple_dataset):
    X, y = simple_dataset
    model = KNNClassifier(k=1).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0)


# --------------------------
# Distance metrics
# --------------------------
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_knn_distance_metrics(metric, random_dataset):
    X, y = random_dataset
    model = KNNClassifier(k=3, dist=metric).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


# --------------------------
# Edge cases / validation
# --------------------------
def test_knn_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        KNNClassifier(k=0)


def test_knn_k_greater_than_samples(simple_dataset):
    X, y = simple_dataset
    with pytest.raises(ValueError, match="is greater than n_samples"):
        KNNClassifier(k=10).fit(X, y)


def test_knn_predict_before_fit(simple_dataset):
    X, _ = simple_dataset
    model = KNNClassifier(k=1)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_knn_invalid_input_shape():
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 1])
    model = KNNClassifier(k=1)
    with pytest.raises(ValueError, match="X must be a 2D array, got shape"):
        model.fit(X, y)


# --------------------------
# Consistency with sklearn
# --------------------------
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_knn_matches_sklearn(metric, random_dataset):
    X, y = random_dataset
    k = 3
    model = KNNClassifier(k=k, dist=metric).fit(X, y)
    if metric == "euclidean":
        sk = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(X, y)
    elif metric == "manhattan":
        sk = KNeighborsClassifier(n_neighbors=k, metric="manhattan").fit(X, y)
    else:  # cosine
        # sklearn doesn't support cosine directly; use metric='cosine'
        sk = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(X, y)

    agreement = np.mean(model.predict(X) == sk.predict(X))
    assert agreement > 0.9  # allow small differences due to tie-breaking
