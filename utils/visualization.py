"""Visualization helpers for deep learning projects."""

import numpy as np


def plot_decision_boundary(pred_func, X, y):
    """Plot an interactive decision boundary using Plotly.

    Parameters
    ----------
    pred_func : callable
        A function that takes an ``(m, 2)`` array and returns predictions.
    X : numpy.ndarray
        Feature matrix, shape ``(m, 2)``.
    y : numpy.ndarray
        Labels, shape ``(m,)``.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object (also calls ``fig.show()``).
    """
    import plotly.graph_objects as go

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(
        data=go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale="Viridis", opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode="markers",
            marker=dict(
                color=y, colorscale="Spectral", size=8,
                line=dict(width=1, color="white"),
            ),
        )
    )
    fig.update_layout(
        title="Decision Boundary",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
    )
    fig.show()
    return fig
