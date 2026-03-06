
import numpy as np


def relu(z):

    return np.maximum(0, z) # ReLU activation function

def relu_grad(z):

    return (z > 0).astype(np.float64) # Gradient of ReLU: 1 for z > 0, else 0

def sigmoid(z):

    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z)) # Sigmoid activation function

def sigmoid_grad(z):

    s = sigmoid(z)
    return s * (1.0 - s) # Gradient of Sigmoid: s * (1 - s)

def tanh(z):

    return np.tanh(z)

def tanh_grad(z):

    return 1.0 - np.tanh(z) ** 2 # Gradient of Tanh: 1 - tanh^2(z)



def softmax(z):

    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True) # For numerical stability: shift values by max to prevent overflow
    e = np.exp(z)

    return e / np.sum(e, axis=1, keepdims=True) 




act_func = {
    "relu":    relu,
    "sigmoid": sigmoid,
    "tanh":    tanh,
}

act_gradeint = {
    "relu":    relu_grad,
    "sigmoid": sigmoid_grad,
    "tanh":    tanh_grad,
}


def get_activation(name):
    #Backward compat helper used by older code

    if name is None:
        return (lambda x: x), (lambda x: np.ones_like(x))
    
    if name not in act_func:
        raise ValueError(f"Unknown activation: {name}")
    
    return act_func[name], act_gradeint[name]