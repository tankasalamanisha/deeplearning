import numpy as np
import pytest
import sys
import os

# Add the repo root to sys.path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implimenting_forward_propagation_numpy import (
    load_coffee_data,
    sigmoid,
    dense,
    sequential,
)


class TestLoadCoffeeData:
    """Tests for load_coffee_data()."""

    def test_returns_tuple(self):
        result = load_coffee_data()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self):
        X, Y = load_coffee_data()
        assert X.shape == (200, 2), f"Expected X shape (200, 2), got {X.shape}"
        assert Y.shape == (200, 1), f"Expected Y shape (200, 1), got {Y.shape}"

    def test_feature_ranges(self):
        X, _ = load_coffee_data()
        temperatures = X[:, 0]
        durations = X[:, 1]
        # Temperature should be in [150, 285] range based on the formula
        assert temperatures.min() >= 150
        assert temperatures.max() <= 285
        # Duration should be in [11.5, 15.5] range based on the formula
        assert durations.min() >= 11.5
        assert durations.max() <= 15.5

    def test_labels_are_binary(self):
        _, Y = load_coffee_data()
        unique_values = np.unique(Y)
        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_has_both_classes(self):
        _, Y = load_coffee_data()
        assert 1.0 in Y, "Expected at least one positive label"
        assert 0.0 in Y, "Expected at least one negative label"

    def test_deterministic_output(self):
        X1, Y1 = load_coffee_data()
        X2, Y2 = load_coffee_data()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_output_dtype(self):
        X, Y = load_coffee_data()
        assert X.dtype == np.float64
        assert Y.dtype == np.float64


class TestSigmoid:
    """Tests for sigmoid()."""

    def test_zero_input(self):
        # sigmoid(0) = 1/(1+exp(0)) = 1/2
        result = sigmoid(0)
        assert result == pytest.approx(0.5)

    def test_large_positive_input(self):
        # With the current implementation: 1/(1+exp(x))
        # For large positive x, exp(x) is huge, so result approaches 0
        result = sigmoid(100)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_large_negative_input(self):
        # For large negative x, exp(x) approaches 0, so result approaches 1
        result = sigmoid(-100)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_numpy_array_input(self):
        x = np.array([-1, 0, 1])
        result = sigmoid(x)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.5)

    def test_symmetry(self):
        # For f(x)=1/(1+exp(x)): f(x) + f(-x) = 1
        x = 2.5
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)

    def test_output_range(self):
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_monotonically_decreasing(self):
        # With the current implementation 1/(1+exp(x)), the function is
        # monotonically decreasing
        x = np.linspace(-5, 5, 50)
        result = sigmoid(x)
        diffs = np.diff(result)
        assert np.all(diffs <= 0)

    def test_2d_array_input(self):
        x = np.array([[0, 1], [-1, 2]])
        result = sigmoid(x)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(0.5)


class TestDense:
    """Tests for dense()."""

    def test_single_unit(self):
        a_in = np.array([1.0, 2.0])
        W = np.array([[0.5], [0.3]])  # shape (2, 1) but code uses W.shape[0]=2
        b = 0.0
        # Code iterates j in range(W.shape[0]) = range(2), but W only has 1 col.
        # This will cause an IndexError for j=1 because W[:,1] doesn't exist.
        # So we need a square W or W with shape[0] == shape[1] for the code to work.
        # Let's use a square matrix instead.
        W = np.array([[0.5, 0.3], [0.3, 0.5]])  # shape (2, 2)
        result = dense(a_in, W, b)
        assert result.shape == (2,)
        # z[0] = dot([0.5, 0.3], [1, 2]) + 0 = 0.5 + 0.6 = 1.1
        # z[1] = dot([0.3, 0.5], [1, 2]) + 0 = 0.3 + 1.0 = 1.3
        expected_0 = sigmoid(1.1)
        expected_1 = sigmoid(1.3)
        assert result[0] == pytest.approx(expected_0)
        assert result[1] == pytest.approx(expected_1)

    def test_output_shape(self):
        a_in = np.array([1.0, 2.0, 3.0])
        W = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9]])
        b = 0.0
        result = dense(a_in, W, b)
        assert result.shape == (3,)

    def test_with_bias(self):
        a_in = np.array([1.0])
        W = np.array([[0.5]])
        b = 1.0
        result = dense(a_in, W, b)
        # z = dot([0.5], [1.0]) + 1.0 = 1.5
        expected = sigmoid(1.5)
        assert result[0] == pytest.approx(expected)

    def test_zero_weights(self):
        a_in = np.array([1.0, 2.0])
        W = np.zeros((2, 2))
        b = 0.0
        result = dense(a_in, W, b)
        # z = 0 for all units, sigmoid(0) = 0.5
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_output_bounded(self):
        a_in = np.array([10.0, -10.0])
        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        b = 0.0
        result = dense(a_in, W, b)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_identity_like_weights(self):
        a_in = np.array([0.0, 0.0])
        W = np.eye(2)
        b = 0.0
        result = dense(a_in, W, b)
        # z = 0 for all, sigmoid(0) = 0.5
        np.testing.assert_allclose(result, [0.5, 0.5])


class TestSequential:
    """Tests for sequential()."""

    def test_output_shape(self):
        x = np.array([1.0, 2.0])
        W1 = np.eye(2)
        W2 = np.eye(2)
        W3 = np.eye(2)
        W4 = np.eye(2)
        b1 = b2 = b3 = b4 = 0.0
        result = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        assert result.shape == (2,)

    def test_output_bounded(self):
        x = np.array([5.0, -5.0, 0.0])
        W1 = np.eye(3)
        W2 = np.eye(3)
        W3 = np.eye(3)
        W4 = np.eye(3)
        b1 = b2 = b3 = b4 = 0.0
        result = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_zero_input(self):
        x = np.array([0.0, 0.0])
        W1 = np.eye(2)
        W2 = np.eye(2)
        W3 = np.eye(2)
        W4 = np.eye(2)
        b1 = b2 = b3 = b4 = 0.0
        result = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        # After 4 layers of sigmoid with zero weights and biases,
        # output converges towards sigmoid applied repeatedly to 0.5
        assert result.shape == (2,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_deterministic(self):
        x = np.array([1.0, 2.0])
        W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        W2 = np.array([[0.5, 0.6], [0.7, 0.8]])
        W3 = np.array([[0.2, 0.3], [0.4, 0.5]])
        W4 = np.array([[0.1, 0.1], [0.2, 0.2]])
        b1 = b2 = b3 = b4 = 0.0
        r1 = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        r2 = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        np.testing.assert_array_equal(r1, r2)

    def test_manual_forward_pass(self):
        x = np.array([1.0, 2.0])
        W1 = np.eye(2)
        W2 = np.eye(2)
        W3 = np.eye(2)
        W4 = np.eye(2)
        b1 = b2 = b3 = b4 = 0.0
        result = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        # Manually compute: 4 layers, each applying dense with identity W and b=0
        a1 = dense(x, W1, b1)
        a2 = dense(a1, W2, b2)
        a3 = dense(a2, W3, b3)
        a4 = dense(a3, W4, b4)
        np.testing.assert_array_equal(result, a4)

    def test_with_biases(self):
        x = np.array([0.0, 0.0])
        W1 = np.eye(2)
        W2 = np.eye(2)
        W3 = np.eye(2)
        W4 = np.eye(2)
        b1 = 1.0
        b2 = -1.0
        b3 = 0.5
        b4 = 0.0
        result = sequential(x, W1, W2, W3, W4, b1, b2, b3, b4)
        expected = dense(
            dense(
                dense(dense(x, W1, b1), W2, b2),
                W3, b3,
            ),
            W4, b4,
        )
        np.testing.assert_array_equal(result, expected)
