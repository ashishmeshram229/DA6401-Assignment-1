import numpy as np


def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float64)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x):
    return np.tanh(np.clip(x, -500, 500))

def tanh_grad(x):
    t = tanh_fn(x)
    return 1.0 - t * t


def identity(x):
    return x

def identity_grad(x):
    return np.ones_like(x, dtype=np.float64)


def get_activation(name):
    name = name.lower()
    if name == "relu":    return relu,     relu_grad
    if name == "sigmoid": return sigmoid,  sigmoid_grad
    if name == "tanh":    return tanh_fn,  tanh_grad
    if name == "none":    return identity, identity_grad
    raise ValueError(f"Unknown activation: {name}")