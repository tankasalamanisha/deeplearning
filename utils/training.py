"""Training loops and prediction helpers for logistic regression."""

import numpy as np

from utils.activations import sigmoid
from utils.costs import binary_cross_entropy
from utils.gradients import compute_gradients
from utils.initialization import initialize_weights


def optimize(W, b, X, Y, alpha, num_iterations, verbose=False):
    """Run gradient descent for logistic regression.

    Parameters
    ----------
    W : numpy.ndarray
        Initial weight vector, shape ``(n, 1)``.
    b : float
        Initial bias.
    X : numpy.ndarray
        Training features, shape ``(n, m)``.
    Y : numpy.ndarray
        Training labels, shape ``(1, m)``.
    alpha : float
        Learning rate.
    num_iterations : int
        Number of gradient descent iterations.
    verbose : bool, optional
        If ``True``, print cost at each iteration.

    Returns
    -------
    W : numpy.ndarray
        Optimized weights.
    b : float
        Optimized bias.
    """
    for i in range(num_iterations):
        Z = np.dot(W.T, X) + b
        A = sigmoid(Z)
        cost = binary_cross_entropy(A, Y)
        if verbose:
            print(f"Iteration {i + 1}/{num_iterations}")
            print(f"Cost: {cost:.4f}")
        dW, db = compute_gradients(X, A, Y)
        W = W - alpha * dW
        b = b - alpha * db
    return W, b


def predict(W, b, X):
    """Predict binary labels using trained logistic regression parameters.

    Parameters
    ----------
    W : numpy.ndarray
        Weight vector, shape ``(n, 1)``.
    b : float
        Bias.
    X : numpy.ndarray
        Input features, shape ``(n, m)``.

    Returns
    -------
    numpy.ndarray
        Predicted labels (0 or 1), shape ``(1, m)``.
    """
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return (A > 0.5).astype(int)


def logistic_regression_model(X_train, Y_train, X_test, Y_test, alpha,
                              num_iterations, verbose=False):
    """Train a logistic regression model and evaluate it.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features, shape ``(n, m_train)``.
    Y_train : numpy.ndarray
        Training labels, shape ``(1, m_train)``.
    X_test : numpy.ndarray
        Test features, shape ``(n, m_test)``.
    Y_test : numpy.ndarray
        Test labels, shape ``(1, m_test)``.
    alpha : float
        Learning rate.
    num_iterations : int
        Number of gradient descent iterations.
    verbose : bool, optional
        If ``True``, print progress information.

    Returns
    -------
    W : numpy.ndarray
        Trained weights.
    b : float
        Trained bias.
    train_accuracy : float
        Training accuracy as a percentage.
    test_accuracy : float
        Test accuracy as a percentage.
    """
    dim = X_train.shape[0]
    W, b = initialize_weights(dim)
    W, b = optimize(W, b, X_train, Y_train, alpha, num_iterations, verbose)

    Y_pred_train = predict(W, b, X_train)
    Y_pred_test = predict(W, b, X_test)
    train_accuracy = np.mean(Y_pred_train == Y_train) * 100
    test_accuracy = np.mean(Y_pred_test == Y_test) * 100

    if verbose:
        print(f"Training accuracy: {train_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")

    return W, b, train_accuracy, test_accuracy
