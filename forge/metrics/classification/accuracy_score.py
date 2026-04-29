import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy as a float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")

    return float(np.mean(y_true == y_pred))
