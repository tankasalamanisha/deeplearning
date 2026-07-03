"""Layer computation utilities."""

import numpy as np

from utils.activations import sigmoid


def dense(a_in, W, b):
    """Compute a fully-connected (dense) layer output.

    Iterates over each unit, computes ``z = w . a_in + b`` and applies
    the sigmoid activation.

    Parameters
    ----------
    a_in : numpy.ndarray
        Activation vector from the previous layer.
    W : numpy.ndarray
        Weight matrix whose columns correspond to individual units.
    b : float or numpy.ndarray
        Bias term.

    Returns
    -------
    numpy.ndarray
        Output activation vector for this layer.
    """
    units = W.shape[0]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b
        a_out[j] = sigmoid(z)
    return a_out


def sequential(x, W1, W2, W3, W4, b1, b2, b3, b4):
    """Forward pass through a four-layer dense network.

    Parameters
    ----------
    x : numpy.ndarray
        Input feature vector.
    W1, W2, W3, W4 : numpy.ndarray
        Weight matrices for layers 1-4.
    b1, b2, b3, b4 : float or numpy.ndarray
        Bias terms for layers 1-4.

    Returns
    -------
    numpy.ndarray
        Output of the final layer.
    """
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    return a4
