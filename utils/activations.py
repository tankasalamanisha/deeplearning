"""Activation functions used across notebooks and scripts."""

import numpy as np


def sigmoid(z):
    """Compute the sigmoid activation function.

    Parameters
    ----------
    z : array_like
        Input value(s).

    Returns
    -------
    numpy.ndarray
        Sigmoid of *z*, element-wise.
    """
    return 1 / (1 + np.exp(-z))
