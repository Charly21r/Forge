from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClassifierMixin
from ..tree.decision_tree import DecisionTreeClassifier
from ..utils.validation import check_array, check_is_fitted, check_X_y


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Random Forest Classifier based on CART Decision Trees"""

    def __init__(
        self,
        *,
        n_estimators: int = 10,
        criterion: str = "gini",
        max_features: str | int | None = "sqrt",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        if isinstance(max_features, str) and max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features must be 'sqrt', 'log2', or the exact number of features")

        self.max_features = max_features
        self.n_estimators = n_estimators
        self.criterion: str = criterion
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.min_impurity_decrease: float = min_impurity_decrease

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> RandomForestClassifier:
        if y is None:
            raise ValueError("y cannot be None for RandomForestClassifier")

        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape[0], X.shape[1]
        self.classes_ = np.unique(y)
        self.trees_ = []

        if self.max_features == "sqrt":
            max_features_count = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features_count = int(np.log2(n_features))
        elif self.max_features is None:
            max_features_count = n_features
        elif isinstance(self.max_features, int):
            max_features_count = self.max_features
        else:
            raise ValueError("max_features must be 'sqrt', 'log2', or the exact number of features")

        if not (1 <= max_features_count <= n_features):
            raise ValueError("max_features must be 'sqrt', 'log2', or the exact number of features")

        for _ in range(self.n_estimators):
            # Sample indices with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = np.random.choice(n_features, size=max_features_count, replace=False)

            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )

            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees_.append((tree, feature_indices))

        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["trees_"])
        X = check_array(X)

        all_preds = np.array(
            [tree.predict(X[:, feature_indices]) for tree, feature_indices in self.trees_]
        )

        return np.array(
            [self.classes_[np.argmax(np.bincount(col.astype(int)))] for col in all_preds.T]
        )

    def predict_proba(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["trees_", "classes_"])
        X = check_array(X)

        n_classes = len(self.classes_)
        accumulated = np.zeros((X.shape[0], n_classes), dtype=float)

        for tree, feature_indices in self.trees_:
            tree_proba = tree.predict_proba(X[:, feature_indices])
            # map tree's local class indices to global self.classes_ positions
            assert tree.classes_ is not None
            for local_idx, cls in enumerate(tree.classes_):
                global_idx = np.searchsorted(self.classes_, cls)
                accumulated[:, global_idx] += tree_proba[:, local_idx]

        return accumulated / len(self.trees_)

    def score(self, X: NDArray[Any], y: NDArray[Any] | None) -> float:
        if y is None:
            raise ValueError("y cannot be None for RandomForestClassifier")

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    @property
    def feature_importances_(self) -> NDArray:
        check_is_fitted(self, ["trees_"])

        n_features = max(fi.max() for _, fi in self.trees_) + 1
        importances = np.zeros(n_features, dtype=float)

        for tree, feature_indices in self.trees_:
            if tree.feature_importances_ is not None:
                for local_idx, global_idx in enumerate(feature_indices):
                    importances[global_idx] += tree.feature_importances_[local_idx]

        total = importances.sum()
        return importances / total if total > 0 else importances
