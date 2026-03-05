import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)


# identity for output layer - returns raw logits as assignment requires
def none_fn(x):
    return x.copy()

def none_deriv(x):
    return np.ones_like(x)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -500, 500))
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


act_map = {
    "sigmoid": (sigmoid,  sigmoid_deriv),
    "tanh":    (tanh_fn,  tanh_deriv),
    "relu":    (relu,     relu_deriv),
    "none":    (none_fn,  none_deriv),
}

def get_activation(name):
    if name not in act_map:
        raise ValueError(f"unknown activation: {name}")
    return act_map[name]
