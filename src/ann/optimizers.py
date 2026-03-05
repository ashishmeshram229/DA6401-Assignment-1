import numpy as np


class BaseOptimizer:
    def update(self, layer, lr, weight_decay):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def __init__(self, lr):
        self.lr = lr

    def update(self, layer, lr_override, weight_decay):
        lr = lr_override if lr_override is not None else self.lr
        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        layer.W -= lr * dW
        layer.b -= lr * dB


class Momentum(BaseOptimizer):
    def __init__(self, lr, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vB   = {}

    def update(self, layer, lr_override, weight_decay):
        lr  = lr_override if lr_override is not None else self.lr
        lid = id(layer)
        if lid not in self.vW:
            self.vW[lid] = np.zeros_like(layer.W)
            self.vB[lid] = np.zeros_like(layer.b)
        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        self.vW[lid] = self.beta * self.vW[lid] + dW
        self.vB[lid] = self.beta * self.vB[lid] + dB
        layer.W -= lr * self.vW[lid]
        layer.b -= lr * self.vB[lid]


class NAG(BaseOptimizer):
    def __init__(self, lr, beta=0.9):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vB   = {}

    def update(self, layer, lr_override, weight_decay):
        lr  = lr_override if lr_override is not None else self.lr
        lid = id(layer)
        if lid not in self.vW:
            self.vW[lid] = np.zeros_like(layer.W)
            self.vB[lid] = np.zeros_like(layer.b)
        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        self.vW[lid] = self.beta * self.vW[lid] + dW
        self.vB[lid] = self.beta * self.vB[lid] + dB
        layer.W -= lr * (self.beta * self.vW[lid] + dW)
        layer.b -= lr * (self.beta * self.vB[lid] + dB)


class RMSProp(BaseOptimizer):
    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self.sW   = {}
        self.sB   = {}

    def update(self, layer, lr_override, weight_decay):
        lr  = lr_override if lr_override is not None else self.lr
        lid = id(layer)
        if lid not in self.sW:
            self.sW[lid] = np.zeros_like(layer.W)
            self.sB[lid] = np.zeros_like(layer.b)
        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        self.sW[lid] = self.beta * self.sW[lid] + (1 - self.beta) * dW ** 2
        self.sB[lid] = self.beta * self.sB[lid] + (1 - self.beta) * dB ** 2
        layer.W -= lr * dW / (np.sqrt(self.sW[lid]) + self.eps)
        layer.b -= lr * dB / (np.sqrt(self.sB[lid]) + self.eps)


def get_optimizer(args):
    name = args.optimizer.lower()
    if name == "sgd":      return SGD(args.learning_rate)
    if name == "momentum": return Momentum(args.learning_rate)
    if name == "nag":      return NAG(args.learning_rate)
    if name == "rmsprop":  return RMSProp(args.learning_rate)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")