import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from forge.exceptions import NotFittedError
from forge.linear import LinearRegression


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def perfect_line():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 3.0 * X.ravel() + 2.0  # y = 3x + 2
    return X, y


@pytest.fixture
def random_dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    coef = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ coef + 0.5 + rng.standard_normal(100) * 0.1
    return X, y


# --------------------------
# Basic functionality
# --------------------------
def test_fit_returns_self(perfect_line):
    X, y = perfect_line
    model = LinearRegression()
    assert model.fit(X, y) is model


def test_predict_perfect_line(perfect_line):
    X, y = perfect_line
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-10)


def test_coef_intercept_perfect_line(perfect_line):
    X, y = perfect_line
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, [3.0], atol=1e-10)
    assert model.intercept_ == pytest.approx(2.0, abs=1e-10)


def test_predict_shape(random_dataset):
    X, y = random_dataset
    preds = LinearRegression().fit(X, y).predict(X)
    assert preds.shape == y.shape


# --------------------------
# fit_intercept=False
# --------------------------
def test_no_intercept_is_zero(perfect_line):
    X, y = perfect_line
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0


def test_no_intercept_coef_shape(random_dataset):
    X, y = random_dataset
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.coef_.shape == (X.shape[1],)


# --------------------------
# Edge cases / validation
# --------------------------
def test_predict_before_fit_raises():
    model = LinearRegression()
    X = np.array([[1.0], [2.0]])
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_y_none_raises():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="y cannot be None"):
        LinearRegression().fit(X, None)


def test_X_y_shape_mismatch_raises():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="y cannot be None for LinearRegression"):
        LinearRegression().fit(X, y)


def test_1d_X_raises():
    X = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2D array"):
        LinearRegression().fit(X, y)


# --------------------------
# Consistency with sklearn
# --------------------------
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_matches_sklearn(random_dataset, fit_intercept):
    X, y = random_dataset
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    sk = SklearnLinearRegression(fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(model.coef_, sk.coef_, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, sk.intercept_, atol=1e-8)
    np.testing.assert_allclose(model.predict(X), sk.predict(X), atol=1e-8)
