import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


ACT_FN = {
    "relu":    relu,
    "sigmoid": sigmoid,
    "tanh":    tanh,
}

ACT_GRAD = {
    "relu":    relu_grad,
    "sigmoid": sigmoid_grad,
    "tanh":    tanh_grad,
}


# kept for backward-compatibility with any code that calls get_activation()
def get_activation(name):
    if name is None:
        return (lambda x: x), (lambda x: np.ones_like(x))
    if name not in ACT_FN:
        raise ValueError(f"Unknown activation: {name}")
    return ACT_FN[name], ACT_GRAD[name]