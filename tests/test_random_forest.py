import numpy as np
import pytest

from forge.ensemble import RandomForestClassifier
from forge.exceptions import NotFittedError


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_fit_returns_self(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=5)
    assert clf.fit(X, y) is clf


def test_fit_sets_classes_and_trees(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=7).fit(X, y)

    assert np.array_equal(clf.classes_, np.array([0, 1]))
    assert len(clf.trees_) == 7
    # each entry is a (tree, feature_indices) pair
    for _tree, feature_indices in clf.trees_:
        assert feature_indices.ndim == 1
        assert feature_indices.shape[0] == int(np.sqrt(X.shape[1]))


def test_predict_shape_and_classes(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_fit_predict_separable():
    np.random.seed(0)
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [10.0, 10.0], [10.1, 10.0], [10.0, 10.2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_estimators=20, max_features=2).fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0, rel=1e-8)


def test_predict_proba_shape_and_sum(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_argmax_matches_predict(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)

    preds = clf.predict(X)
    proba_preds = clf.classes_[np.argmax(clf.predict_proba(X), axis=1)]
    assert np.array_equal(preds, proba_preds)


def test_predict_before_fit_raises():
    clf = RandomForestClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(np.array([[1.0, 2.0]]))


def test_predict_proba_before_fit_raises():
    clf = RandomForestClassifier()
    with pytest.raises(NotFittedError):
        clf.predict_proba(np.array([[1.0, 2.0]]))


def test_fit_with_none_y_raises():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    clf = RandomForestClassifier()
    with pytest.raises(ValueError, match="y cannot be None for RandomForestClassifier"):
        clf.fit(X, None)


def test_invalid_max_features_string_in_init():
    with pytest.raises(
        ValueError, match="max_features must be 'sqrt', 'log2', or the exact number of features"
    ):
        RandomForestClassifier(max_features="invalid")


def test_invalid_max_features_int_too_large():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    clf = RandomForestClassifier(max_features=10)
    with pytest.raises(
        ValueError, match="max_features must be 'sqrt', 'log2', or the exact number of features"
    ):
        clf.fit(X, y)


def test_invalid_max_features_int_zero():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    clf = RandomForestClassifier(max_features=0)
    with pytest.raises(
        ValueError, match="max_features must be 'sqrt', 'log2', or the exact number of features"
    ):
        clf.fit(X, y)


@pytest.mark.parametrize("max_features", ["sqrt", "log2", None, 2])
def test_max_features_variants(max_features):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (X[:, 0] > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=5, max_features=max_features).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (40,)


def test_feature_importances_shape_and_normalized(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=10, max_features=None).fit(X, y)

    importances = clf.feature_importances_
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    assert np.isclose(importances.sum(), 1.0)


def test_feature_importances_before_fit_raises():
    clf = RandomForestClassifier()
    with pytest.raises(NotFittedError):
        _ = clf.feature_importances_


def test_score_matches_accuracy(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    assert clf.score(X, y) == pytest.approx(float(np.mean(clf.predict(X) == y)))


def test_score_with_none_y_raises(simple_data):
    X, y = simple_data
    clf = RandomForestClassifier(n_estimators=5).fit(X, y)
    with pytest.raises(ValueError, match="y cannot be None for RandomForestClassifier"):
        clf.score(X, None)


def test_overfits_training_data():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=25, max_features=3).fit(X, y)
    assert clf.score(X, y) > 0.95
