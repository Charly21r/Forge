from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base.base_estimator import BaseEstimator
from ..base.mixins import ClassifierMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class Node:
    feature: int | None = None
    threshold: Any | None = None
    left: Node | None = None
    right: Node | None = None
    depth: int = 0
    n_samples: int = 0
    impurity: float = 0.0
    is_categorical: bool = False
    proba: np.ndarray | None = None
    label: int | None = None
    subtree_leaves: int = 1
    subtree_risk: float = 0.0
    alpha_node: float = np.inf
    node_risk: float = 0.0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """CART Decision Tree implementation from scratch."""

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.criterion: str = criterion
        self.max_depth: int | None = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.min_impurity_decrease: float = min_impurity_decrease

        self.classes_: NDArray[Any] | None = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.n_samples_total_: int = 0
        self.tree: Node | None = None
        self.feature_importances_: NDArray[Any] | None = None

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None) -> DecisionTreeClassifier:
        if y is None:
            raise ValueError("y cannot be None for DecisionTreeClassifier")

        X, y = check_X_y(X, y)

        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.n_samples_total_ = X.shape[0]

        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)

        idx = np.arange(X.shape[0])
        self.tree = self._build(X, y_enc, idx, depth=0)

        self._compute_feature_importances(self.tree)
        self._normalize_feature_importances()

        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, ["tree", "classes_"])
        X = check_array(X)

        proba: NDArray[np.float64] = self.predict_proba(X)
        y_pred_idx: NDArray[np.int64] = np.argmax(proba, axis=1)

        assert self.classes_ is not None
        classes: NDArray[np.int64] = self.classes_
        return classes[y_pred_idx]

    def predict_proba(self, X: NDArray[Any]) -> np.ndarray:
        check_is_fitted(self, ["tree", "classes_"])
        X = check_array(X)
        out = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i in range(X.shape[0]):
            node = self.tree
            while node is not None and not node.is_leaf():
                if node.is_categorical:
                    if (
                        node.feature is not None
                        and node.threshold is not None
                        and X[i, node.feature] in node.threshold
                    ):
                        node = node.left
                    else:
                        node = node.right
                else:
                    if (
                        node.feature is not None
                        and node.threshold is not None
                        and X[i, node.feature] < node.threshold
                    ):
                        node = node.left
                    else:
                        node = node.right
            if node is not None and node.proba is not None:
                out[i] = node.proba
            else:
                out[i] = np.zeros(self.n_classes_, dtype=float)
        return out

    def score(self, X: NDArray[Any], y: NDArray[Any] | None) -> float:
        if y is None:
            raise ValueError("y cannot be None for DecisionTreeClassifier")

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def _gini(self, counts: NDArray[np.int_]) -> float:
        total: int = int(counts.sum())
        if total == 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.sum(p * p))

    def _entropy(self, counts: NDArray[np.int_]) -> float:
        total: int = int(counts.sum())
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _impurity(self, counts: NDArray[np.int_]) -> float:
        if self.criterion == "gini":
            return self._gini(counts)
        return self._entropy(counts)

    def _best_split_one_feature_numerical(
        self, x: NDArray[Any], y: NDArray[Any]
    ) -> tuple[float | None, float]:
        x = np.asarray(x)
        y = np.asarray(y)
        order = np.argsort(x, kind="mergesort")
        x = x[order]
        classes, y_enc = np.unique(y[order], return_inverse=True)
        K = len(classes)
        N = len(x)
        if N <= 1:
            return None, float(np.inf)
        parent_counts = np.bincount(y_enc, minlength=self.n_classes_)
        prefix = np.zeros((N + 1, K), dtype=int)
        for i in range(1, N + 1):
            prefix[i] = prefix[i - 1]
            prefix[i, y_enc[i - 1]] += 1
        best_t: float | None = None
        best_after_imp: float = float(np.inf)
        for i in range(1, N):
            if x[i - 1] == x[i]:
                continue
            t = (x[i - 1] + x[i]) / 2
            n_left = i
            n_right = N - i
            left_counts = prefix[i]
            right_counts = parent_counts - left_counts
            if n_left == 0 or n_right == 0:
                continue
            left_imp = self._impurity(left_counts)
            right_imp = self._impurity(right_counts)
            weighted_after = (n_left / N) * left_imp + (n_right / N) * right_imp
            if weighted_after < best_after_imp:
                best_after_imp = weighted_after
                best_t = t
        return best_t, best_after_imp

    def _best_split_one_feature_categorical(
        self, x: NDArray[Any], y: NDArray[Any]
    ) -> tuple[NDArray[Any] | None, float]:
        cats, x_enc = np.unique(x, return_inverse=True)
        K = len(cats)
        if K <= 1:
            return None, float(np.inf)
        best_subset = None
        best_after_imp = float(np.inf)
        for r in range(1, K):
            for subset in combinations(range(K), r):
                mask_left = np.isin(x_enc, subset)
                left_counts = np.bincount(y[mask_left], minlength=self.n_classes_)
                right_counts = np.bincount(y[~mask_left], minlength=self.n_classes_)
                n_left = mask_left.sum()
                n_right = (~mask_left).sum()
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_imp = self._impurity(left_counts)
                right_imp = self._impurity(right_counts)
                N = n_left + n_right
                weighted_after = (n_left / N) * left_imp + (n_right / N) * right_imp
                if weighted_after < best_after_imp:
                    best_after_imp = weighted_after
                    best_subset = subset
        if best_subset is None:
            return None, float(np.inf)
        return cats[list(best_subset)], best_after_imp

    def _best_split_one_feature(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[Any | None, float]:
        if np.issubdtype(np.asarray(x).dtype, np.number):
            return self._best_split_one_feature_numerical(x, y)
        return self._best_split_one_feature_categorical(x, y)

    def _best_split(self, X: NDArray[Any], y: NDArray[Any]) -> tuple[int | None, Any | None, float]:
        X = np.asarray(X)
        y = np.asarray(y)
        _N, D = X.shape
        best_feat: int | None = None
        best_t: Any | None = None
        best_imp: float = float(np.inf)
        for i in range(D):
            t, imp = self._best_split_one_feature(X[:, i], y)
            if t is None:
                continue
            if imp < best_imp:
                best_imp = imp
                best_feat = i
                best_t = t
        return best_feat, best_t, best_imp

    def _build(
        self, X: NDArray[Any], y: NDArray[Any], idx: NDArray[np.int_], depth: int = 0
    ) -> Node:
        node = Node(depth=depth, n_samples=idx.size)
        counts = np.bincount(y[idx], minlength=self.n_classes_)
        node.proba = (
            counts / counts.sum() if counts.sum() > 0 else np.zeros(self.n_classes_, dtype=float)
        )
        node.label = (
            int(np.argmax(node.proba)) if node.proba is not None and node.proba.size > 0 else None
        )
        node.impurity = self._impurity(counts)
        n_misclassified = counts.sum() - counts.max() if counts.size > 0 else 0
        node.node_risk = (
            float(n_misclassified) / float(self.n_samples_total_)
            if self.n_samples_total_ > 0
            else 0.0
        )
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or idx.size < self.min_samples_split
            or counts.max() == counts.sum()
        ):
            node.subtree_leaves = 1
            node.subtree_risk = node.node_risk
            return node
        best_feat, best_t, best_imp = self._best_split(X[idx], y[idx])
        if best_feat is None:
            node.subtree_leaves = 1
            node.subtree_risk = node.node_risk
            return node
        gain = node.impurity - best_imp
        if gain < self.min_impurity_decrease:
            node.subtree_leaves = 1
            node.subtree_risk = node.node_risk
            return node
        if np.issubdtype(X[idx, best_feat].dtype, np.number):
            mask_left = X[idx, best_feat] <= best_t
        else:
            if best_t is not None:
                mask_left = np.isin(X[idx, best_feat], np.asarray(best_t))
            else:
                mask_left = np.zeros_like(X[idx, best_feat], dtype=bool)
            node.is_categorical = True
        left_idx = idx[mask_left]
        right_idx = idx[~mask_left]
        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            node.subtree_leaves = 1
            node.subtree_risk = node.node_risk
            return node
        node.feature = int(best_feat)
        node.threshold = best_t
        node.left = self._build(X, y, left_idx, depth + 1)
        node.right = self._build(X, y, right_idx, depth + 1)
        node.subtree_leaves = (node.left.subtree_leaves if node.left else 0) + (
            node.right.subtree_leaves if node.right else 0
        )
        node.subtree_risk = (node.left.subtree_risk if node.left else 0.0) + (
            node.right.subtree_risk if node.right else 0.0
        )
        if node.subtree_leaves - 1 == 0:
            node.alpha_node = np.inf
        else:
            node.alpha_node = (node.node_risk - node.subtree_risk) / (node.subtree_leaves - 1)
        return node

    def _update_subtree_stats(self, node: Node | None) -> None:
        if node is None:
            return
        if node.is_leaf():
            node.subtree_leaves = 1
            node.subtree_risk = node.node_risk
            node.alpha_node = np.inf
            return
        self._update_subtree_stats(node.left)
        self._update_subtree_stats(node.right)
        node.subtree_leaves = (node.left.subtree_leaves if node.left else 0) + (
            node.right.subtree_leaves if node.right else 0
        )
        node.subtree_risk = (node.left.subtree_risk if node.left else 0.0) + (
            node.right.subtree_risk if node.right else 0.0
        )
        if node.subtree_leaves - 1 == 0:
            node.alpha_node = np.inf
        else:
            node.alpha_node = (node.node_risk - node.subtree_risk) / (node.subtree_leaves - 1)

    def _find_min_alpha_node(self, node: Node | None) -> tuple[Node | None, float]:
        if node is None or node.is_leaf():
            return None, float(np.inf)
        best_node: Node | None = node
        best_alpha: float = node.alpha_node
        left_node, left_alpha = self._find_min_alpha_node(node.left)
        right_node, right_alpha = self._find_min_alpha_node(node.right)
        if left_alpha < best_alpha:
            best_node = left_node
            best_alpha = left_alpha
        if right_alpha < best_alpha:
            best_node = right_node
            best_alpha = right_alpha
        return best_node, best_alpha

    def prune(self, ccp_alpha: float = 0.0) -> None:
        if self.tree is None:
            return
        while True:
            self._update_subtree_stats(self.tree)
            min_node, min_alpha = self._find_min_alpha_node(self.tree)
            if min_node is None or min_alpha > ccp_alpha:
                break
            min_node.left = None
            min_node.right = None
            min_node.alpha_node = np.inf
            min_node.subtree_leaves = 1
            min_node.subtree_risk = min_node.node_risk

    def _compute_feature_importances(self, node: Node | None) -> None:
        if node is None or node.is_leaf():
            return
        if node.left is None or node.right is None:
            return
        impurity_left = node.left.impurity * node.left.n_samples
        impurity_right = node.right.impurity * node.right.n_samples
        impurity_decrease = node.impurity * node.n_samples - impurity_left - impurity_right
        if node.feature is not None and node.feature < self.n_features_:
            assert self.feature_importances_ is not None
            self.feature_importances_[node.feature] += impurity_decrease
        self._compute_feature_importances(node.left)
        self._compute_feature_importances(node.right)

    def _normalize_feature_importances(self) -> None:
        if self.feature_importances_ is None:
            return
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total

    def print_tree(self) -> None:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        self._print_node(self.tree)

    def _print_node(self, node: Node | None, indent: str = "") -> None:
        if node is None:
            return
        if node.is_leaf():
            proba_str = np.round(node.proba, 3).tolist() if node.proba is not None else []
            print(f"{indent}Leaf(depth={node.depth}, n={node.n_samples}, proba={proba_str})")
            return
        if node.is_categorical:
            print(f"{indent}[X{node.feature} in {node.threshold}] imp={node.impurity:.3f}")
        else:
            print(f"{indent}[X{node.feature} < {node.threshold:.3f}] imp={node.impurity:.3f}")
        if node.left is not None:
            self._print_node(node.left, indent + "  ")
        if node.right is not None:
            self._print_node(node.right, indent + "  ")
