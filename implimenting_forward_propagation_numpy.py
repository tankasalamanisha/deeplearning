import numpy as np


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5          # 12-15 min is best
    X[:,0] = X[:,0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1,1))


def sigmoid(x):
    x = np.clip(x, -500, 500)
    output = 1/(1+np.exp(-x))
    return output


def dense(a_in, W, b):
    """Compute a layer output using the activation vector from previous layer, weights and b.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If input dimensions are incompatible with weight matrix.
    """
    if not isinstance(a_in, np.ndarray):
        raise TypeError(f"a_in must be a numpy array, got {type(a_in).__name__}")
    if not isinstance(W, np.ndarray):
        raise TypeError(f"W must be a numpy array, got {type(W).__name__}")
    if W.ndim != 2:
        raise ValueError(f"W must be a 2D array, got {W.ndim}D")
    if a_in.shape[0] != W.shape[0]:
        raise ValueError(
            f"Shape mismatch: a_in has {a_in.shape[0]} elements "
            f"but W has {W.shape[0]} rows"
        )

    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b
        a_out[j] = sigmoid(z)

    return a_out


def sequential(x, W1, W2, W3, W4, b1, b2, b3, b4):
    arrays = {"x": x, "W1": W1, "W2": W2, "W3": W3, "W4": W4}
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")

    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    f_x = a4

    return f_x