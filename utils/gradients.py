"""Gradient computation helpers for logistic regression."""

import numpy as np


def compute_gradients(X, A, Y):
    """Compute gradients for logistic regression (vectorized).

    Parameters
    ----------
    X : numpy.ndarray
        Input features, shape ``(n, m)``.
    A : numpy.ndarray
        Predicted probabilities, shape ``(1, m)``.
    Y : numpy.ndarray
        True labels, shape ``(1, m)``.

    Returns
    -------
    dW : numpy.ndarray
        Gradient of cost w.r.t. weights, shape ``(n, 1)``.
    db : float
        Gradient of cost w.r.t. bias.
    """
    m = Y.shape[1]
    dZ = A - Y
    dW = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    return dW, db
