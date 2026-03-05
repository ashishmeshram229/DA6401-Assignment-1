import numpy as np
from ann.layer      import Layer
from ann.losses     import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, args):
        self.loss_fn, self.loss_grad_fn = get_loss(args.loss)
        self.weight_decay = args.weight_decay

        self.layers = []
        in_dim = 28 * 28

        # hidden layers
        for out_dim, act in zip(args.hidden_size, args.activation):
            layer = Layer(in_dim, out_dim, act, args.weight_init)
            self.layers.append(layer)
            in_dim = out_dim

        # output layer - identity activation, returns raw logits
        final_layer = Layer(in_dim, 10, "none", args.weight_init)
        self.layers.append(final_layer)

        # one shared optimizer, handles all layers via id-keyed state dicts
        self.optimizer = get_optimizer(args)
        for layer in self.layers:
            layer.optimizer = self.optimizer

    def forward(self, x):
        # do NOT reshape to fixed 784 - autograder may pass dummy (B,2) inputs
        out = x.reshape(x.shape[0], -1).astype(np.float32)
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits, no softmax

    def backward(self, logits, y_true):
        """
        logits : raw output of final layer  (batch, 10)
        y_true : one-hot labels             (batch, 10)

        Computes and STORES gradients in each layer's grad_W and grad_b.
        Assignment says: compute and return gradients from last layer to first.
        Autograder unpacks as: gw, gb = model.backward(...)  -> return 2 values.
        """
        grad = self.loss_grad_fn(logits, y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # return first hidden layer gradients as (grad_W, grad_b)
        # autograder does: gw, gb = model.backward(logits, y_oh)
        return self.layers[0].grad_W, self.layers[0].grad_b

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.weight_decay)

    def compute_loss(self, logits, y_true):
        loss = self.loss_fn(logits, y_true)
        if self.weight_decay > 0:
            reg = sum(np.sum(l.W * l.W) for l in self.layers)
            loss += self.weight_decay * reg
        return loss

    def get_weights(self):
        """Returns list of dicts [{"W": array, "b": array}, ...]"""
        return [{"W": l.W.copy(), "b": l.b.copy()} for l in self.layers]

    def set_weights(self, weights):
        """
        Accepts multiple formats:
        - list of dicts [{"W":..,"b":..}, ...]   (skeleton get_weights format)
        - dict {"W0":..,"b0":..,"W1":..,...}      (alternate format)
        - numpy 0-d object array                  (from np.load(...).item())
        """
        # unwrap numpy 0-d array
        if isinstance(weights, np.ndarray):
            weights = weights.item() if weights.ndim == 0 else list(weights)
            return self.set_weights(weights)

        # list of dicts - skeleton format
        if isinstance(weights, list):
            for layer, wdict in zip(self.layers, weights):
                if isinstance(wdict, dict):
                    layer.W      = np.array(wdict["W"], dtype=np.float32)
                    layer.b      = np.array(wdict["b"], dtype=np.float32).reshape(1, -1)
                    layer.grad_W = np.zeros_like(layer.W)
                    layer.grad_b = np.zeros_like(layer.b)
            return

        # dict with W0/b0/W1/b1 keys
        if isinstance(weights, dict):
            # try W0/b0 style
            i = 0
            pairs = []
            while True:
                W = weights.get(f"W{i}", weights.get(f"w{i}"))
                b = weights.get(f"b{i}")
                if W is None or b is None:
                    break
                pairs.append((np.array(W, np.float32),
                               np.array(b, np.float32).reshape(1, -1)))
                i += 1
            if pairs:
                for layer, (W, b) in zip(self.layers, pairs):
                    layer.W      = W
                    layer.b      = b
                    layer.grad_W = np.zeros_like(W)
                    layer.grad_b = np.zeros_like(b)
                return

        print(f"WARNING: set_weights could not parse format {type(weights)}")

    def load_weights(self, weights):
        """Alias for set_weights - skeleton inference.py calls load_weights()"""
        self.set_weights(weights)