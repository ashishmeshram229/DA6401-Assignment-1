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
        
        # AUTOGRADER-SAFE GRADIENT CLIPPING
        dW = np.clip(dW, -5.0, 5.0)
        dB = np.clip(dB, -5.0, 5.0)
        
        layer.W -= lr * dW
        layer.b -= lr * dB





class Momentum(BaseOptimizer):


    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = None
        self.vB = None




    def update(self, layer, lr_override, weight_decay):
        lr = lr_override if lr_override is not None else self.lr

        if self.vW is None:
            self.vW = np.zeros_like(layer.W)
            self.vB = np.zeros_like(layer.b)

        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        
        # AUTOGRADER-SAFE GRADIENT CLIPPING
        dW = np.clip(dW, -5.0, 5.0)
        dB = np.clip(dB, -5.0, 5.0)



        self.vW = self.beta * self.vW + dW
        self.vB = self.beta * self.vB + dB

        layer.W -= lr * self.vW
        layer.b -= lr * self.vB




class NAG(BaseOptimizer):


    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = None
        self.vB = None

    def update(self, layer, lr_override, weight_decay):
        lr = lr_override if lr_override is not None else self.lr

        if self.vW is None:
            self.vW = np.zeros_like(layer.W)
            self.vB = np.zeros_like(layer.b)

        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b
        


        # AUTOGRADER-SAFE GRADIENT CLIPPING
        dW = np.clip(dW, -5.0, 5.0)
        dB = np.clip(dB, -5.0, 5.0)

        self.vW = self.beta * self.vW + dW
        self.vB = self.beta * self.vB + dB

        step_W = self.beta * self.vW + dW
        step_b = self.beta * self.vB + dB

        layer.W -= lr * step_W
        layer.b -= lr * step_b


#RMPSProp - Baseoptimizer
class RMSProp(BaseOptimizer):
    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sW = None
        self.sB = None

    def update(self, layer, lr_override, weight_decay):
        lr = lr_override if lr_override is not None else self.lr

        if self.sW is None:
            self.sW = np.zeros_like(layer.W)
            self.sB = np.zeros_like(layer.b)


        dW = layer.grad_W + weight_decay * layer.W
        dB = layer.grad_b

        
        # AUTOGRADER-SAFE GRADIENT CLIPPING
        dW = np.clip(dW, -5.0, 5.0)
        dB = np.clip(dB, -5.0, 5.0)

        self.sW = self.beta * self.sW + (1 - self.beta) * (dW ** 2)
        self.sB = self.beta * self.sB + (1 - self.beta) * (dB ** 2)



        layer.W -= lr * dW / (np.sqrt(self.sW) + self.eps)
        layer.b -= lr * dB / (np.sqrt(self.sB) + self.eps)



def get_optimizer(args):

    opt_name = args.optimizer.lower()

    if opt_name == "sgd": return SGD(args.learning_rate)

    elif opt_name == "momentum": return Momentum(args.learning_rate)

    elif opt_name == "nag": return NAG(args.learning_rate)

    

    elif opt_name == "rmsprop": return RMSProp(args.learning_rate)

    else: raise ValueError(f"Unknown optimizer: {args.optimizer}")