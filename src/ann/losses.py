import numpy as np


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def log_softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x), axis=1, keepdims=True))
    return x - logsumexp


def cross_entropy_loss(logits, y_true):
    log_probs = log_softmax(logits)
    return -np.sum(y_true * log_probs) / logits.shape[0]

def cross_entropy_grad(logits, y_true):
    # returns (probs - y_true) / m  -- already divided by m
    # so layer.backward must NOT divide by m again
    probs = softmax(logits)
    return (probs - y_true) / logits.shape[0]


def mse_loss(logits, y_true):
    diff = logits - y_true
    return np.mean(diff ** 2)

def mse_grad(logits, y_true):
    m = logits.shape[0]
    n = logits.shape[1]
    return (2.0 / (m * n)) * (logits - y_true)


def get_loss(name):
    name = name.lower()
    # support both "mse" and "mean_squared_error" for CLI compatibility
    if name == "cross_entropy":              return cross_entropy_loss, cross_entropy_grad
    if name in ("mse", "mean_squared_error"): return mse_loss, mse_grad
    raise ValueError(f"Unknown loss: {name}")