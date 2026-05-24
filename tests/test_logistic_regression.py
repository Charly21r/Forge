import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from forge.exceptions import NotFittedError
from forge.linear import LogisticRegression


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def linearly_separable():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
    return X, y


@pytest.fixture
def noisy_binary():
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    return X, y


# --------------------------
# Basic functionality
# --------------------------
def test_fit_returns_self(linearly_separable):
    X, y = linearly_separable
    model = LogisticRegression()
    assert model.fit(X, y) is model


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    preds = LogisticRegression().fit(X, y).predict(X)
    assert preds.shape == y.shape


def test_perfect_separation_high_accuracy(linearly_separable):
    X, y = linearly_separable
    model = LogisticRegression(max_iter=2000).fit(X, y)
    acc = float(np.mean(model.predict(X) == y))
    assert acc >= 0.98


def test_predict_proba_sums_to_one(linearly_separable):
    X, y = linearly_separable
    proba = LogisticRegression().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_predict_proba_in_unit_interval(linearly_separable):
    X, y = linearly_separable
    proba = LogisticRegression().fit(X, y).predict_proba(X)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_predict_proba_shape(linearly_separable):
    X, y = linearly_separable
    proba = LogisticRegression().fit(X, y).predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


def test_predict_consistent_with_proba(linearly_separable):
    X, y = linearly_separable
    model = LogisticRegression().fit(X, y)
    proba = model.predict_proba(X)
    preds = model.predict(X)
    expected = model.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(preds, expected)


# --------------------------
# Classes / labels
# --------------------------
def test_classes_attribute(linearly_separable):
    X, y = linearly_separable
    model = LogisticRegression().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_arbitrary_class_labels():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 2))
    y = np.where(X[:, 0] > 0, "pos", "neg")
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({"pos", "neg"})
    assert float(np.mean(preds == y)) >= 0.9


def test_more_than_two_classes_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 2))
    y = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError, match="binary classification"):
        LogisticRegression().fit(X, y)


# --------------------------
# fit_intercept
# --------------------------
def test_no_intercept_is_zero(linearly_separable):
    X, y = linearly_separable
    model = LogisticRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0


# --------------------------
# Edge cases / validation
# --------------------------
def test_predict_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0]])
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_proba_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0]])
    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_y_none_raises():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="y cannot be None"):
        LogisticRegression().fit(X, None)


def test_X_y_shape_mismatch_raises():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="same number of samples"):
        LogisticRegression().fit(X, y)


# --------------------------
# Consistency with sklearn
# --------------------------
def test_matches_sklearn_direction(noisy_binary):
    """Our GD-trained model should agree with sklearn on the decision
    boundary direction (cosine similarity of coefs close to 1)."""
    X, y = noisy_binary
    ours = LogisticRegression(max_iter=5000, C=1.0).fit(X, y)
    sk = SklearnLogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(X, y)
    cos = float(
        ours.coef_ @ sk.coef_.ravel() / (np.linalg.norm(ours.coef_) * np.linalg.norm(sk.coef_))
    )
    assert cos > 0.99


def test_matches_sklearn_accuracy(noisy_binary):
    X, y = noisy_binary
    ours = LogisticRegression(max_iter=5000).fit(X, y)
    sk = SklearnLogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
    ours_acc = float(np.mean(ours.predict(X) == y))
    sk_acc = float(np.mean(sk.predict(X) == y))
    assert abs(ours_acc - sk_acc) < 0.02


# --------------------------
# Regularization
# --------------------------
def test_smaller_C_shrinks_coefficients(noisy_binary):
    X, y = noisy_binary
    weak = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    strong = LogisticRegression(C=0.01, max_iter=5000).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)
