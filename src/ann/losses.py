import numpy as np


def softmax(x):
    x  = np.asarray(x, dtype=np.float64)
    x  = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def cross_entropy_loss(logits, y_true):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    probs  = np.clip(softmax(logits), 1e-12, 1.0)
    m      = logits.shape[0]
    return float(-np.sum(y_true * np.log(probs)) / m)

def cross_entropy_grad(logits, y_true):
    # returns (probs - y) / m  — divides by m once
    # layer.backward must NOT divide by m again
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    m      = logits.shape[0]
    return (softmax(logits) - y_true) / m


def mse_loss(logits, y_true):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((logits - y_true) ** 2))

def mse_grad(logits, y_true):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    m, n   = logits.shape
    return (2.0 / (m * n)) * (logits - y_true)


def get_loss(name):
    name = name.lower()
    if name == "cross_entropy":               return cross_entropy_loss, cross_entropy_grad
    if name in ("mse", "mean_squared_error"): return mse_loss,          mse_grad
    raise ValueError(f"Unknown loss: {name}")