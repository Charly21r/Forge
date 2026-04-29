"""
forge — Handbuilt ML library in pure NumPy.

sklearn-compatible machine learning algorithms in pure NumPy.
"""

from .neighbors import KNNClassifier
from .tree import DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier", "KNNClassifier"]
