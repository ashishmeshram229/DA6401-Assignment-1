import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.activations import act_func, act_gradient


class Layer:
    def __init__(self, in_dim, out_dim, activation, weight_init): # activation can be None for output layer (e.g. softmax is applied separately in loss)
        self.activation = activation

        self._init_weights(in_dim, out_dim, weight_init)

       
        self.grad_W = np.zeros((in_dim, out_dim), dtype=np.float64)

        self.grad_b = np.zeros((1,      out_dim), dtype=np.float64)

        self._input = None
        self._z     = None



    @property
    def grad_w(self): 
        return self.grad_W 



    @grad_w.setter
    def grad_w(self, v):
        self.grad_W = v

    def _init_weights(self, in_dim, out_dim, method): # weight initialization method: "xavier" or "random"

        if method == "xavier":
            limit  = np.sqrt(6.0 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit,
                                       (in_dim, out_dim)).astype(np.float64)
            
        else:   # random
            self.W = (np.random.randn(in_dim, out_dim) * 0.01).astype(np.float64)

        self.b = np.zeros((1, out_dim), dtype=np.float64)

    def forward(self, x):

        # Store input and pre-activation values for use in backward pass
        self._input = np.asarray(x, dtype=np.float64)
        self._z     = self._input @ self.W + self.b

        if self.activation is None:
            return self._z
        

        return act_func[self.activation](self._z)

    def backward(self, delta):
       
        delta = np.asarray(delta, dtype=np.float64)

        if self.activation is not None: # If there's an activation function, we need to apply the chain rule to compute the gradient w.r.t. pre-activation values
            delta = delta * act_gradient[self.activation](self._z)

        self.grad_W = self._input.T @ delta
        self.grad_b = np.sum(delta, axis=0, keepdims=True) # Gradient w.r.t. biases is the sum of deltas across the batch


        return delta @ self.W.T 