import numpy as np

from src.gradient_descent import (
    normalize_features,
    predict,
    compute_loss,
    compute_gradients,
    train_gradient_descent,
)


def test_normalize_features_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_norm, mean, std = normalize_features(X)
    assert X_norm.shape == X.shape
    assert np.allclose(mean, [2.0, 3.0])
    assert np.all(std > 0)


def test_predict_shape():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([0.5, 0.3])
    b = 1.0
    y_pred = predict(X, w, b)
    assert y_pred.shape == (2,)


def test_compute_loss_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    loss = compute_loss(y, y_pred)
    assert np.isclose(loss, 0.0)


def test_compute_gradients_zero_at_minimum():
    X = np.array([[1.0], [2.0], [3.0]])
    w = np.array([2.0])
    b = 1.0
    y = predict(X, w, b)
    y_pred = predict(X, w, b)
    grad_w, grad_b = compute_gradients(X, y, y_pred)
    assert np.allclose(grad_w, 0.0)
    assert np.isclose(grad_b, 0.0)


def test_train_gradient_descent_learns_simple_line():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2 * X.flatten() + 1
    X_norm, _, _ = normalize_features(X)
    w, b = train_gradient_descent(X_norm, y, learning_rate=0.1, n_iterations=500)
    y_pred = predict(X_norm, w, b)
    loss = compute_loss(y, y_pred)
    assert loss < 1.0
