import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer



class NeuralNetwork:

    def __init__(self, args):
        
        if not hasattr(args, 'hidden_size') and hasattr(args, 'sz'):
            args.hidden_size = args.sz
        
        self.loss_fn, self.loss_grad_fn = get_loss(args.loss)
        self.layers = []
        input_dim = 28 * 28  # MNIST/Fashion-MNIST dimension

        # Hidden layers
        for out_dim, act in zip(args.hidden_size, args.activation):

            layer = Layer(input_dim, out_dim, act, args.weight_init)
            self.layers.append(layer)
            input_dim = out_dim

        # Final output layer (returns logits, no activation)

        final_layer = Layer(input_dim, 10, "none", args.weight_init)
        self.layers.append(final_layer)


        # Assign optimizer per layer

        self.optimizer = get_optimizer(args)

        self.weight_decay = args.weight_decay

        for layer in self.layers:

            layer.optimizer = get_optimizer(args)

    # Forward Pass
    
    def forward(self, x):
        out = x.reshape(x.shape[0], -1).astype(np.float64)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # Backward Pass
    def backward(self, logits, y_true):
       
        grad = self.loss_grad_fn(logits, y_true)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


    # Update Parameters

    
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.weight_decay)


    # Compute Full 
    
    def compute_loss(self, logits, y_true):
        loss = self.loss_fn(logits, y_true)
        if self.weight_decay > 0:
            reg = 0.0
            for layer in self.layers:
                reg += np.sum(layer.W * layer.W)
            loss += self.weight_decay * reg
        return loss



    # Autograder Load/Save Weight Methods
    def get_weights(self):

        weights = []
        for layer in self.layers:

            weights.append({
                "W": layer.W.copy(),
                "b": layer.b.copy()
            })
        return weights



    def set_weights(self, weights_list):


        for layer, wdict in zip(self.layers, weights_list):

            layer.W = wdict["W"].copy()
            layer.b = wdict["b"].copy()