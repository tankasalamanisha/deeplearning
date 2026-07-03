"""Cost / loss functions used across notebooks and scripts."""

import numpy as np


def binary_cross_entropy(A, Y):
    """Compute the average binary cross-entropy cost.

    Parameters
    ----------
    A : numpy.ndarray
        Predicted probabilities, shape ``(1, m)``.
    Y : numpy.ndarray
        True labels (0 or 1), shape ``(1, m)``.

    Returns
    -------
    float
        Average cost over all *m* examples.
    """
    m = Y.shape[1]
    cost = -(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return np.sum(cost) / m
