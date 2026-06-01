import numpy as np
import pytest

from forge.tree import DecisionTreeRegressor

# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def linear_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return X, y


@pytest.fixture
def two_group_data():
    """Two clearly separated groups — depth-1 tree should find perfect split."""
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([1.0, 3.0, 7.0, 9.0])
    return X, y


# ------------------------------------------------------------------ #
# Basic correctness                                                    #
# ------------------------------------------------------------------ #


def test_fit_returns_self(linear_data):
    X, y = linear_data
    reg = DecisionTreeRegressor()
    assert reg.fit(X, y) is reg


def test_predict_output_shape(linear_data):
    X, y = linear_data
    reg = DecisionTreeRegressor().fit(X, y)
    assert reg.predict(X).shape == (len(y),)


def test_predict_output_dtype(linear_data):
    X, y = linear_data
    reg = DecisionTreeRegressor().fit(X, y)
    assert reg.predict(X).dtype == float


def test_not_fitted_raises():
    reg = DecisionTreeRegressor()
    with pytest.raises(RuntimeError):
        reg.predict(np.array([[1.0]]))


def test_y_none_raises():
    reg = DecisionTreeRegressor()
    with pytest.raises(ValueError, match="y cannot be None for DecisionTreeRegressor"):
        reg.fit(np.array([[1.0]]), None)


# ------------------------------------------------------------------ #
# Leaf values                                                          #
# ------------------------------------------------------------------ #


def test_mse_leaf_value_is_mean(two_group_data):
    X, y = two_group_data
    reg = DecisionTreeRegressor(criterion="mse", max_depth=1).fit(X, y)
    preds = reg.predict(X)
    # left group mean = (1+3)/2 = 2.0, right group mean = (7+9)/2 = 8.0
    assert preds[0] == pytest.approx(2.0)
    assert preds[2] == pytest.approx(8.0)


def test_mae_leaf_value_is_median():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([1.0, 2.0, 9.0, 4.0, 5.0, 6.0])
    reg = DecisionTreeRegressor(criterion="mae", max_depth=1).fit(X, y)
    preds = reg.predict(X)
    # left median = 2.0, right median = 5.0
    assert preds[0] == pytest.approx(2.0)
    assert preds[3] == pytest.approx(5.0)


# ------------------------------------------------------------------ #
# Overfitting / depth control                                          #
# ------------------------------------------------------------------ #


def test_overfit_memorizes_training_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    y = X[:, 0] * 2 + X[:, 1]

    reg = DecisionTreeRegressor(max_depth=None).fit(X, y)
    assert reg.score(X, y) == pytest.approx(1.0, abs=1e-6)


def test_max_depth_limits_accuracy(linear_data):
    X, y = linear_data
    shallow = DecisionTreeRegressor(max_depth=1).fit(X, y)
    deep = DecisionTreeRegressor(max_depth=None).fit(X, y)
    assert deep.score(X, y) >= shallow.score(X, y)


def test_score_is_r2(two_group_data):
    X, y = two_group_data
    reg = DecisionTreeRegressor(max_depth=1).fit(X, y)
    preds = reg.predict(X)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    expected_r2 = 1 - ss_res / ss_tot
    assert reg.score(X, y) == pytest.approx(expected_r2)


# ------------------------------------------------------------------ #
# Hyperparameter constraints                                           #
# ------------------------------------------------------------------ #


def test_min_samples_split_respected():
    X = np.arange(10).reshape(-1, 1).astype(float)
    y = np.arange(10).astype(float)

    reg_constrained = DecisionTreeRegressor(min_samples_split=10).fit(X, y)
    reg_free = DecisionTreeRegressor(min_samples_split=2).fit(X, y)

    # constrained tree cannot split → single leaf → worse or equal score
    assert reg_free.score(X, y) >= reg_constrained.score(X, y)


def test_invalid_criterion_raises():
    with pytest.raises(ValueError, match="criterion must be"):
        DecisionTreeRegressor(criterion="variance")


# ------------------------------------------------------------------ #
# Feature importances                                                  #
# ------------------------------------------------------------------ #


def test_feature_importances_shape():
    X = np.random.randn(30, 3)
    y = X[:, 0] * 2.0
    reg = DecisionTreeRegressor().fit(X, y)
    assert reg.feature_importances_.shape == (3,)


def test_feature_importances_sum_to_one():
    X = np.random.randn(30, 3)
    y = X[:, 0] * 2.0
    reg = DecisionTreeRegressor().fit(X, y)
    assert np.isclose(reg.feature_importances_.sum(), 1.0)


def test_most_important_feature_is_correct():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 3))
    y = X[:, 1] * 5.0  # feature 1 drives y
    reg = DecisionTreeRegressor().fit(X, y)
    assert np.argmax(reg.feature_importances_) == 1


# ------------------------------------------------------------------ #
# Pruning                                                              #
# ------------------------------------------------------------------ #


def test_pruning_reduces_leaves():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20).astype(float)

    reg = DecisionTreeRegressor().fit(X, y)
    leaves_before = reg.tree.subtree_leaves

    reg.prune(ccp_alpha=0.5)
    assert reg.tree.subtree_leaves <= leaves_before


def test_pruning_alpha_zero_does_not_change_predictions():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 2))
    y = X[:, 0] + X[:, 1]

    reg = DecisionTreeRegressor().fit(X, y)
    preds_before = reg.predict(X)
    reg.prune(ccp_alpha=0.0)
    assert np.allclose(preds_before, reg.predict(X))
