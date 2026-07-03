"""Data loading and preprocessing utilities."""

import numpy as np


def load_coffee_data():
    """Create a synthetic coffee roasting data set.

    Optimal roasting parameters:
    - Duration: 12-15 minutes
    - Temperature: 175-260 C

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape ``(200, 2)`` with columns
        ``[temperature, duration]``.
    Y : numpy.ndarray
        Labels of shape ``(200, 1)`` — 1 for good roast, 0 otherwise.
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5            # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150   # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if t > 175 and t < 260 and d > 12 and d < 15 and d <= y:
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


def normalize_images(images, scale=255.0):
    """Normalize image pixel values to the [0, 1] range.

    Parameters
    ----------
    images : numpy.ndarray
        Raw image array with pixel values in [0, *scale*].
    scale : float, optional
        Maximum pixel value (default 255.0).

    Returns
    -------
    numpy.ndarray
        Normalized image array with values in [0, 1].
    """
    return images / scale
