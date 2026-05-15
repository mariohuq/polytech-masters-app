import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BoundsDetector(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible detector for multi-dimensional time series.
    Checks if values in each dimension fall outside of user-defined ranges.

    Parameters
    ----------
    bounds : dict, optional
        A dictionary defining the valid range for each dimension.
        Format: {'dimension_name': (low_threshold, high_threshold)}
        - low_threshold: Minimum inclusive value.
        - high_threshold: Maximum inclusive value.
        Dimensions not included in this dictionary will default to
        (-infinity, +infinity) and will never trigger an out-of-bounds flag.

        Note: Order of features is induced from fit, not from bounds keys!
    """

    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else {}
        self.feature_names_in_ = None
        self.low_vec_ = None
        self.high_vec_ = None

    def fit(self, X, y=None):
        """
        Prepares the threshold vectors based on input data dimensions.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The training data to determine feature count and names.
        """
        # Capture feature count for validation
        self.n_features_in_ = X.shape[1]

        # Initialize vectors with 'infinite' bounds.
        # This ensures columns without specific rules remain "In Bounds" (0).
        self.low_vec_ = np.full(self.n_features_in_, -np.inf)
        self.high_vec_ = np.full(self.n_features_in_, np.inf)

        # Map the 'bounds' dictionary keys to the correct vector indices
        if hasattr(X, 'columns'):
            # Case 1: X is a pandas DataFrame (Map by Name)
            self.feature_names_in_ = list(X.columns)
            for name, (low, high) in self.bounds.items():
                if name in self.feature_names_in_:
                    idx = self.feature_names_in_.index(name)
                    self.low_vec_[idx] = low
                    self.high_vec_[idx] = high
        else:
            # Case 2: X is a numpy array (Map by Position)
            # We assume the order of keys in the dict matches column order
            for i, (name, (low, high)) in enumerate(self.bounds.items()):
                if i < self.n_features_in_:
                    self.low_vec_[i] = low
                    self.high_vec_[i] = high

        return self

    def predict(self, X):
        """
        Performs a vectorized out-of-bounds check across all dimensions.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The input time series data.

        Returns
        -------
        out : ndarray of shape (n_samples, n_features)
            Binary matrix where 1 = Out of Bounds and 0 = In Bounds.
        """
        check_is_fitted(self)
        X_val = check_array(X)

        if X_val.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: fit on {self.n_features_in_}, "
                             f"got {X_val.shape[1]}")

        # --- Vectorized Logic Explained ---
        # NumPy Broadcasting allows us to compare a 2D matrix (Samples x Dims)
        # against a 1D vector (Dims).
        # (X_val < self.low_vec_) results in a boolean matrix of the same shape as X.

        lower_mask = X_val < self.low_vec_
        upper_mask = X_val > self.high_vec_

        # Combine masks: True if either condition is met
        out_of_bounds = lower_mask | upper_mask

        return out_of_bounds.astype(int)
