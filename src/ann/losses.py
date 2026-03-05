import numpy as np
from ann.activations import softmax


# IMPORTANT convention used throughout this codebase:
# loss_grad functions do NOT divide by m (batch size)
# layer.backward() is the ONLY place that divides by m
# this avoids double-division and keeps gradient check correct

def cross_entropy_loss(logits, y_oh):
    probs = softmax(logits)
    m     = y_oh.shape[0]
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(y_oh * np.log(probs)) / m)

def cross_entropy_grad(logits, y_oh):
    # d(CE)/d(logits) before softmax = probs - y_oh
    # no /m here - layer.backward divides by m
    probs = softmax(logits)
    return probs - y_oh


def mse_loss(logits, y_oh):
    probs = softmax(logits)
    return float(np.mean((probs - y_oh) ** 2))

def mse_grad(logits, y_oh):
    # chain rule through softmax for mse
    # no /m here - layer.backward divides by m
    probs  = softmax(logits)
    n      = y_oh.shape[1]
    dl_dp  = 2.0 * (probs - y_oh) / n
    s      = np.sum(dl_dp * probs, axis=1, keepdims=True)
    return probs * (dl_dp - s)


def get_loss(name):
    if name == "cross_entropy":
        return cross_entropy_loss, cross_entropy_grad
    elif name == "mean_squared_error":
        return mse_loss, mse_grad
    else:
        raise ValueError(f"unknown loss: {name}")
