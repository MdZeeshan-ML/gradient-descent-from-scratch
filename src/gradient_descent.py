from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance.

    Args:
        X: Raw feature matrix of shape (n_samples, n_features).

    Returns:
        X_norm: Normalized features.
        mean: Per-feature mean used for normalization.
        std: Per-feature standard deviation used for normalization.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    return X_norm, mean, std


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute predictions for linear regression model.

    Args:
        X: Feature matrix (n_samples, n_features).
        w: Weight vector (n_features,).
        b: Bias term.

    Returns:
        Predicted targets of shape (n_samples,).
    """
    return X @ w + b


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error loss.

    Args:
        y_true: Ground-truth targets (n_samples,).
        y_pred: Predicted targets (n_samples,).

    Returns:
        Scalar MSE loss.
    """
    n = y_true.shape[0]
    return float(np.sum((y_pred - y_true) ** 2) / n)


def compute_gradients(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute analytical gradients of MSE w.r.t. weights and bias.

    Args:
        X: Feature matrix (n_samples, n_features).
        y_true: Ground-truth targets (n_samples,).
        y_pred: Predicted targets (n_samples,).

    Returns:
        grad_w: Gradient w.r.t. weights.
        grad_b: Gradient w.r.t. bias.
    """
    n = y_true.shape[0]
    error = y_pred - y_true
    grad_w = (X.T @ error) / n
    grad_b = float(error.sum() / n)
    return grad_w, grad_b


def train_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    n_iterations: int,
) -> Tuple[np.ndarray, float]:
    """Train linear regression using batch gradient descent.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        learning_rate: Step size for gradient descent.
        n_iterations: Number of update steps.

    Returns:
        w: Learned weights.
        b: Learned bias.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(n_iterations):
        y_pred = predict(X, w, b)
        grad_w, grad_b = compute_gradients(X, y, y_pred)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b
