import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

from forge.naive_bayes import GaussianNB


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def simple_dataset():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def random_dataset():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 3, 100)
    return X, y


# --------------------------
# Basic functionality
# --------------------------
def test_gnb_fit_predict_simple(simple_dataset):
    X, y = simple_dataset
    model = GaussianNB().fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.mean(preds == y) > 0.9


def test_gnb_predict_proba_shape(simple_dataset):
    X, y = simple_dataset
    model = GaussianNB().fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (len(X), len(np.unique(y)))


def test_gnb_predict_proba_sums_to_one(simple_dataset):
    X, y = simple_dataset
    model = GaussianNB().fit(X, y)

    probs = model.predict_proba(X)

    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


# --------------------------
# Edge cases / validation
# --------------------------
def test_gnb_predict_before_fit():
    model = GaussianNB()
    X = np.array([[1.0], [2.0]])

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_gnb_invalid_input_shape():
    X = np.array([1.0, 2.0, 3.0])  # not 2D
    y = np.array([0, 1, 1])

    model = GaussianNB()

    with pytest.raises(ValueError, match="X must be a 2D array"):
        model.fit(X, y)


def test_gnb_single_class():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])

    model = GaussianNB().fit(X, y)
    preds = model.predict(X)

    assert np.all(preds == 0)


# --------------------------
# Numerical stability
# --------------------------
def test_gnb_zero_variance_feature():
    # All samples identical: variance = 0
    X = np.array([[1.0], [1.0], [1.0], [10.0], [10.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GaussianNB().fit(X, y)
    probs = model.predict_proba(X)

    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()


# --------------------------
# Consistency with sklearn
# --------------------------
def test_gnb_matches_sklearn(random_dataset):
    X, y = random_dataset

    model = GaussianNB().fit(X, y)
    sk_model = SklearnGaussianNB().fit(X, y)

    preds = model.predict(X)
    sk_preds = sk_model.predict(X)

    agreement = np.mean(preds == sk_preds)

    assert agreement > 0.9  # allow small numerical differences


def test_gnb_predict_proba_close_to_sklearn(random_dataset):
    X, y = random_dataset

    model = GaussianNB().fit(X, y)
    sk_model = SklearnGaussianNB().fit(X, y)

    probs = model.predict_proba(X)
    sk_probs = sk_model.predict_proba(X)

    assert np.allclose(probs, sk_probs, atol=1e-1)
