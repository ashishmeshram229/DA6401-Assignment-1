import numpy as np


# Activation Functions and their gradients for backpropagation



# ReLU

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float64)

# Sigmoid 

def sigmoid(x):
    return np.where(x >= 0, 
                    1.0 / (1.0 + np.exp(-x)), 
                    np.exp(x) / (1.0 + np.exp(x)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# Tanh
def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    t = np.tanh(x)
    return 1.0 - t * t

# Identity (for final layer logits)

def identity(x):
    return x

def identity_grad(x):
    return np.ones_like(x)

# Activation factory
def get_activation(name):
    
    name = name.lower()
    if name == "relu":
        return relu, relu_grad
    

    elif name == "sigmoid":
        return sigmoid, sigmoid_grad
    
    elif name == "tanh":
        return tanh, tanh_grad
    
    elif name == "none":
        return identity, identity_grad
    else:
        raise ValueError(f"Unknown activation: {name}")