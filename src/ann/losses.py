import numpy as np


# Utility: Stable softmax and log-softmax

def softmax(x):
    
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def log_softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x_shifted), axis=1, keepdims=True))
    return x_shifted - logsumexp

# Cross Entropy Loss

def cross_entropy_loss(logits, y_true):
   
    log_probs = log_softmax(logits)
    return -np.sum(y_true * log_probs) / logits.shape[0]

def cross_entropy_grad(logits, y_true):
   
    probs = softmax(logits)
    return (probs - y_true) / logits.shape[0]

# Mean Squared Error (ON LOGITS)

def mse_loss(logits, y_true):

    diff = logits - y_true

    return np.mean(diff ** 2)

def mse_grad(logits, y_true):

    m = logits.shape[0]
    n = logits.shape[1]

    return (2.0 / (m * n)) * (logits - y_true)

# Loss

def get_loss(name):

    name = name.lower()
    
    if name == "cross_entropy":
        return cross_entropy_loss, cross_entropy_grad
    
    elif name == "mean_squared_error":
        return mse_loss, mse_grad
    
    else:
        raise ValueError(f"Unknown loss: {name}")