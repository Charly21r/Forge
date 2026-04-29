import numpy as np
import pytest

from forge.tree import DecisionTreeClassifier


def test_fit_predict_numerical():
    X = np.array([[0.0], [0.1], [0.2], [10.0]])
    y = np.array([0, 0, 0, 1])

    # Use min_samples_leaf=1 to allow split for single sample
    clf = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
    preds = clf.predict(X)

    # exact predictions
    assert np.array_equal(preds, y)

    # score correctness
    assert clf.score(X, y) == pytest.approx(1.0, rel=1e-8)


def test_categorical_split_and_feature_importances():
    X = np.array([["red"], ["red"], ["blue"], ["blue"]], dtype=object)
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier().fit(X, y)

    # tree structure
    assert clf.tree is not None
    assert clf.tree.is_categorical is True

    # feature importance normalization
    assert clf.feature_importances_.shape == (1,)
    assert np.isclose(clf.feature_importances_.sum(), 1.0)

    # perfect prediction
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_pruning_reduces_tree_complexity():
    X = np.array([[1], [2], [3], [4], [6], [7], [8], [9], [2.5]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

    clf = DecisionTreeClassifier(min_samples_split=2).fit(X, y)
    leaves_before = clf.tree.subtree_leaves

    clf.prune(ccp_alpha=0.2)
    leaves_after = clf.tree.subtree_leaves

    assert leaves_after <= leaves_before


def test_print_tree_does_not_raise(capsys):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    clf = DecisionTreeClassifier().fit(X, y)
    clf.print_tree()

    # capture output (ensures something was printed)
    captured = capsys.readouterr()
    assert "Leaf" in captured.out


@pytest.fixture
def simple_tree_data():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_tree_basic(simple_tree_data):
    X, y = simple_tree_data
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    preds = model.predict(X)

    assert np.mean(preds == y) == 1.0


def test_tree_overfit():
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=None).fit(X, y)
    preds = model.predict(X)

    assert np.mean(preds == y) > 0.95


def test_tree_not_fitted():
    model = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))


def test_tree_output_shape():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)

    model = DecisionTreeClassifier().fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (20,)


def test_tree_invariant_prediction_after_pruning_when_alpha_zero():
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)

    clf = DecisionTreeClassifier().fit(X, y)
    preds_before = clf.predict(X)

    clf.prune(ccp_alpha=0.0)
    preds_after = clf.predict(X)

    assert np.array_equal(preds_before, preds_after)
