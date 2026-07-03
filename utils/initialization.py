"""Weight initialization helpers."""

import numpy as np


def initialize_weights(n):
    """Initialize weights and bias for a single-layer model.

    Parameters
    ----------
    n : int
        Number of input features.

    Returns
    -------
    W : numpy.ndarray
        Weight vector of shape ``(n, 1)`` with small random values.
    b : float
        Bias, initialized to 0.
    """
    W = np.random.randn(n, 1) * 0.01
    b = 0
    return W, b
