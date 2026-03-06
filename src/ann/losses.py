import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ann.activations import softmax


def cross_entropy(logits, y_true): # Compute cross-entropy loss given raw logits and true class labels (as integers)
    
    probs = softmax(logits)
    n     = y_true.shape[0]


    return float(np.mean(-np.log(probs[np.arange(n), y_true.astype(int)] + 1e-9)))


def cross_entropy_grad(logits, y_true):


    probs = softmax(logits) # Compute gradient of cross-entropy loss w.r.t. logits: grad = (probs - one_hot) / n
    n     = y_true.shape[0]
    probs[np.arange(n), y_true.astype(int)] -= 1.0
    return probs / n


def mse(logits, y_true): # Compute mean squared error loss given raw logits and true class labels (as integers)
    probs   = softmax(logits)

    n, c    = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true.astype(int)] = 1.0



    return float(np.mean((probs - one_hot) ** 2))



def mse_grad(logits, y_true):


    probs   = softmax(logits)
    n, c    = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true.astype(int)] = 1.0
    diff    = probs - one_hot
    grad    = np.zeros_like(probs) # Gradient of MSE loss w.r.t. logits



    for k in range(c): # Compute gradient for each class k using the formula: grad[:, k] = (2/c) * sum(diff * dsm) where dsm is the Jacobian of softmax for class k
        dsm        = probs * (np.eye(c)[k] - probs[:, k:k+1])
        grad[:, k] = np.sum((2.0 / c) * diff * dsm, axis=1)
    return grad / n


loss_func = {
    "cross_entropy":      cross_entropy,
    "mse":                mse,
    "mean_squared_error": mse,
}

loss_gradient = {
    "cross_entropy":      cross_entropy_grad,
    "mse":                mse_grad,
    "mean_squared_error": mse_grad,
}