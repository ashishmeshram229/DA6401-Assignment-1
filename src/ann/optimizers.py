import numpy as np


class BaseOptimizer:
    def update(self, layer, lr, weight_decay):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def update(self, layer, lr, weight_decay):
        layer.W -= lr * (layer.grad_W + weight_decay * layer.W)
        layer.b -= lr * layer.grad_b


class Momentum(BaseOptimizer):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def update(self, layer, lr, weight_decay):
        lid = id(layer)
        if lid not in self.vW:
            self.vW[lid] = np.zeros_like(layer.W)
            self.vb[lid] = np.zeros_like(layer.b)
        gW = layer.grad_W + weight_decay * layer.W
        self.vW[lid] = self.beta * self.vW[lid] + gW
        self.vb[lid] = self.beta * self.vb[lid] + layer.grad_b
        layer.W -= lr * self.vW[lid]
        layer.b -= lr * self.vb[lid]


class NAG(BaseOptimizer):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def update(self, layer, lr, weight_decay):
        lid = id(layer)
        if lid not in self.vW:
            self.vW[lid] = np.zeros_like(layer.W)
            self.vb[lid] = np.zeros_like(layer.b)
        gW = layer.grad_W + weight_decay * layer.W
        self.vW[lid] = self.beta * self.vW[lid] + gW
        self.vb[lid] = self.beta * self.vb[lid] + layer.grad_b
        layer.W -= lr * (self.beta * self.vW[lid] + gW)
        layer.b -= lr * (self.beta * self.vb[lid] + layer.grad_b)


class RMSProp(BaseOptimizer):
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps  = eps
        self.sW   = {}
        self.sb   = {}

    def update(self, layer, lr, weight_decay):
        lid = id(layer)
        if lid not in self.sW:
            self.sW[lid] = np.zeros_like(layer.W)
            self.sb[lid] = np.zeros_like(layer.b)
        gW = layer.grad_W + weight_decay * layer.W
        gb = layer.grad_b
        self.sW[lid] = self.beta * self.sW[lid] + (1 - self.beta) * gW ** 2
        self.sb[lid] = self.beta * self.sb[lid] + (1 - self.beta) * gb ** 2
        layer.W -= lr * gW / (np.sqrt(self.sW[lid]) + self.eps)
        layer.b -= lr * gb / (np.sqrt(self.sb[lid]) + self.eps)


def get_optimizer(args):
    name = args.optimizer.lower()
    if name == "sgd":      return SGD()
    if name == "momentum": return Momentum()
    if name == "nag":      return NAG()
    if name == "rmsprop":  return RMSProp()
    raise ValueError(f"Unknown optimizer: {args.optimizer}")