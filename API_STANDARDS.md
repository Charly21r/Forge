# API Design Standards

This document outlines the standardized API design patterns used throughout Forge, modeled after scikit-learn's proven design.

## Core Principles

### 1. Fit-Predict Pattern

All estimators follow this pattern:

```python
from forge import DecisionTreeClassifier
import numpy as np

# Create estimator
clf = DecisionTreeClassifier(max_depth=5)

# Fit to training data
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Get probability predictions (for classifiers)
y_proba = clf.predict_proba(X_test)

# Score on test data
accuracy = clf.score(X_test, y_test)
```

### 2. Estimator Interface Requirements

Every estimator must implement:

```python
class MyEstimator:
    # Configuration in __init__
    def __init__(self, param1=default1, param2=default2):
        """Initialize estimator with parameters."""
        pass
    
    # Learning
    def fit(self, X, y) -> "MyEstimator":
        """Fit model to data. Returns self for chaining."""
        pass
    
    # Prediction
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    # Evaluation
    def score(self, X, y) -> float:
        """Return evaluation metric (accuracy for classifiers)."""
        pass
    
    # Parameter access
    def get_params(self, deep=True) -> dict:
        """Get estimator parameters."""
        pass
    
    def set_params(self, **params) -> "MyEstimator":
        """Set estimator parameters."""
        pass
    
    # Representation
    def __repr__(self) -> str:
        """String representation with all parameters."""
        pass
```

### 3. Input Validation

All estimators must validate inputs:

```python
def fit(self, X, y):
    """Fit the estimator."""
    X, y = check_X_y(X, y)
    
    # Additional specific validations
    if self.param < 0:
        raise ValueError(f"param must be non-negative, got {self.param}")
    
    return self
```

### 4. Attributes Naming

Attributes follow strict naming conventions:

- **Parameters**: `self.param_name` (as given to `__init__`)
- **Fitted Attributes**: `self.param_name_` (ending with underscore)
  - `self.classes_` - unique class labels
  - `self.n_classes_` - number of classes
  - `self.n_features_` - number of features
  - `self.n_samples_` - number of training samples
  - `self.feature_importances_` - feature importance scores
  - `self.tree_` - fitted tree structure

Example:

```python
class MyEstimator:
    def __init__(self, max_depth=None):
        # Input parameter (no underscore)
        self.max_depth = max_depth
    
    def fit(self, X, y):
        # Fitted attributes (with underscore)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        return self
```

### 5. Error Handling

Use custom exceptions for specific errors:

```python
from forge.exceptions import NotFittedError, InvalidMetricError

# Check if fitted
def _check_fitted(self):
    if not hasattr(self, "classes_") or self.classes_ is None:
        raise NotFittedError(
            f"{self.__class__.__name__} is not fitted. "
            f"Call 'fit' before using this method."
        )

# Specific exceptions
def __init__(self, metric="euclidean"):
    if metric not in {"euclidean", "manhattan", "cosine"}:
        raise InvalidMetricError(
            f"Unknown metric '{metric}'. "
            f"Allowed metrics: euclidean, manhattan, cosine"
        )
```

### 6. Docstring Template

All public methods must follow this structure:

```python
def method_name(self, param1: Type1, param2: Type2 = default) -> ReturnType:
    """Short one-line description.

    Longer description explaining what the method does and why.
    Can span multiple lines if needed.

    Parameters
    ----------
    param1 : Type1
        Description of param1.
    param2 : Type2, default=default
        Description of param2.

    Returns
    -------
    result : ReturnType
        Description of returned value.

    Raises
    ------
    ExceptionType
        When exception is raised and why.

    Examples
    --------
    >>> from forge import MyEstimator
    >>> import numpy as np
    >>> X = np.array([[0], [1], [2]])
    >>> y = np.array([0, 1, 1])
    >>> est = MyEstimator()
    >>> est.fit(X, y)
    MyEstimator()
    >>> est.predict(np.array([[1.5]]))
    array([1])

    Notes
    -----
    Additional technical notes or references.

    See Also
    --------
    OtherEstimator : Related estimator.
    """
```

### 7. Type Hints

Always use complete type hints:

```python
from typing import Optional, Union, Tuple
from numpy.typing import NDArray
import numpy as np

class Estimator:
    def fit(
        self, 
        X: NDArray, 
        y: NDArray,
        sample_weight: Optional[NDArray] = None
    ) -> "Estimator":
        """Fit estimator."""
        ...

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, any]:
        """Get parameters."""
        ...
```

### 8. Method Chaining

Methods that modify state should return `self`:

```python
# Good: allows chaining
clf = DecisionTreeClassifier(max_depth=5).fit(X, y).score(X, y)

class Estimator:
    def fit(self, X, y) -> "Estimator":
        """..."""
        return self  # Always return self!
    
    def set_params(self, **params) -> "Estimator":
        """..."""
        return self  # Always return self!
```

### 9. Parameter Validation Template

```python
class Estimator:
    def __init__(self, param1: int = 5, param2: str = "gini"):
        # Type hints in __init__ signature
        
        # Validate enums
        if param2 not in {"gini", "entropy"}:
            raise ValueError(
                f"param2 must be 'gini' or 'entropy', got {param2}"
            )
        
        # Validate numeric ranges
        if param1 <= 0:
            raise ValueError(
                f"param1 must be positive, got {param1}"
            )
        
        # Store validated parameters
        self.param1 = param1
        self.param2 = param2
```

### 10. Logging

Use logging for informational messages:

```python
import logging

logger = logging.getLogger(__name__)

class Estimator:
    def fit(self, X, y):
        logger.info(
            f"Fitting {self.__class__.__name__} "
            f"on {X.shape[0]} samples with {X.shape[1]} features"
        )
        
        # ... actual fitting code ...
        
        logger.debug(f"Classes found: {self.classes_}")
        logger.debug(f"Feature importances: {self.feature_importances_}")
        
        return self
```

## Implementation Checklist

- [ ] Inherit from `BaseEstimator`
- [ ] Parameters in `__init__` with validation
- [ ] Complete type hints on all methods
- [ ] Full docstrings in NumPy format
- [ ] `fit()` returns `self`
- [ ] `predict()` raises `NotFittedError` if not fitted
- [ ] `get_params()` and `set_params()` implemented
- [ ] `__repr__()` shows all parameters
- [ ] Input validation in `fit()` using `check_X_y()`
- [ ] Fitted attributes end with `_`
- [ ] Logging for key operations
- [ ] Custom exceptions for specific errors
- [ ] Comprehensive tests with edge cases
- [ ] Examples in docstrings

## Extending the API

### Adding a New Estimator

1. Create file: `forge/module/estimator.py`
2. Inherit from `BaseEstimator`
3. Implement all required methods
4. Add to `forge/module/__init__.py`
5. Add to main `forge/__init__.py`
6. Create corresponding test file
7. Update documentation

### Adding a New Parameter

1. Add to `__init__` with validation
2. Store as `self.param_name`
3. Use in `fit()` or `predict()`
4. Update docstring
5. Add test cases for parameter validation

## Testing Standards

All public methods must have tests covering:
- Basic functionality
- Input validation
- Edge cases
- Error handling
- Consistency with sklearn (where applicable)

```python
class TestEstimator:
    def test_fit_predict(self):
        """Test basic fit and predict."""
        est = Estimator()
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        assert preds.shape == y_test.shape
    
    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            Estimator().fit(invalid_X, y)
    
    def test_not_fitted_error(self):
        """Test NotFittedError."""
        from forge.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            Estimator().predict(X_test)
    
    def test_parameter_validation(self):
        """Test parameter validation in __init__."""
        with pytest.raises(ValueError):
            Estimator(invalid_param=value)
```

## References

- [scikit-learn API Design](https://scikit-learn.org/stable/developers/develop.html)
- [NumPy Docstring Format](https://numpydoc.readthedocs.io/en/latest/format.html)
- [PEP 257 - Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
