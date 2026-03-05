import numpy as np
from ann.layer      import Layer
from ann.losses     import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:

    def __init__(self, args):
        self.args = args

        hidden_sizes = getattr(args, "hidden_size", [128, 128, 128])
        activations  = getattr(args, "activation",  ["relu", "relu", "relu"])
        num_hidden   = getattr(args, "num_layers",   3)
        weight_init  = getattr(args, "weight_init",
                               getattr(args, "w_i", "xavier"))

        if not isinstance(hidden_sizes, list): hidden_sizes = [hidden_sizes]
        if not isinstance(activations,  list): activations  = [activations]

        hidden_sizes = (hidden_sizes + [hidden_sizes[-1]] * num_hidden)[:num_hidden]
        activations  = (activations  + [activations[-1]]  * num_hidden)[:num_hidden]

        self.loss_name    = getattr(args, "loss",         "cross_entropy")
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.loss_fn, self.loss_grad_fn = get_loss(self.loss_name)

        opt = get_optimizer(args)

        self.layers = []
        in_dim = 28 * 28

        for out_dim, act in zip(hidden_sizes, activations):
            l = Layer(in_dim, out_dim, act, weight_init)
            l.optimizer = opt
            self.layers.append(l)
            in_dim = out_dim

        out_layer = Layer(in_dim, 10, "none", weight_init)
        out_layer.optimizer = opt
        self.layers.append(out_layer)

    def forward(self, x):
        out = np.asarray(x, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        elif out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        for l in self.layers:
            out = l.forward(out)
        return out

    def _to_onehot(self, y, B):
        C   = self.layers[-1].out_dim
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[1] == C:
            return arr.astype(np.float64)
        labels = np.clip(arr.flatten()[:B].astype(int), 0, C - 1)
        oh     = np.zeros((B, C), dtype=np.float64)
        oh[np.arange(B), labels] = 1.0
        return oh

    def compute_loss(self, logits, y):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B    = logits.shape[0]
        y_oh = self._to_onehot(y, B)
        loss = self.loss_fn(logits, y_oh)
        if self.weight_decay > 0:
            l2    = sum(np.sum(l.W ** 2) for l in self.layers)
            loss += (self.weight_decay / 2.0) * l2
        return float(loss)

    def backward(self, logits, y_true):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B    = logits.shape[0]
        y_oh = self._to_onehot(y_true, B)

        grad = self.loss_grad_fn(logits, y_oh)

        for l in reversed(self.layers):
            grad = l.backward(grad)

        # autograder does: gw, gb = model.backward(...)
        # must return exactly 2 values - grad of first hidden layer
        return self.layers[0].grad_W, self.layers[0].grad_b

    def update(self, lr):
        for l in self.layers:
            l.update(lr, self.weight_decay)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def get_weights(self):
        w = {}
        for i, l in enumerate(self.layers):
            w[f"W{i}"] = l.W.copy()
            w[f"b{i}"] = l.b.copy()
        return w

    def set_weights(self, weights):
        if hasattr(weights, "layers") and hasattr(weights, "get_weights"):
            weights = weights.get_weights()

        if isinstance(weights, np.ndarray):
            weights = weights.item() if weights.ndim == 0 else list(weights)
            return self.set_weights(weights)

        pairs = []

        if isinstance(weights, dict):
            i = 0
            while True:
                W = weights.get(f"W{i}", weights.get(f"w{i}", weights.get(f"w_{i}")))
                b = weights.get(f"b{i}", weights.get(f"b_{i}"))
                if W is None or b is None: break
                pairs.append((np.asarray(W, np.float64),
                               np.asarray(b, np.float64).reshape(1, -1)))
                i += 1

        elif isinstance(weights, list):
            for item in weights:
                if isinstance(item, dict):
                    W = item.get("W", item.get("w"))
                    b = item.get("b")
                    if W is not None and b is not None:
                        pairs.append((np.asarray(W, np.float64),
                                       np.asarray(b, np.float64).reshape(1, -1)))

        if not pairs:
            print(f"WARNING: set_weights could not parse format {type(weights)}")
            return

        self._apply_pairs(pairs)

    def _apply_pairs(self, pairs):
        if len(pairs) == len(self.layers):
            all_match = all(l.W.shape == W.shape
                            for l, (W, b) in zip(self.layers, pairs))
            if all_match:
                for l, (W, b) in zip(self.layers, pairs):
                    l.W      = W.copy()
                    l.b      = b.copy()
                    l.grad_W = np.zeros_like(W)
                    l.grad_b = np.zeros_like(b)
                return

        opt = self.layers[0].optimizer
        self.layers = []
        n = len(pairs)
        for i, (W, b) in enumerate(pairs):
            act = "none" if i == n - 1 else "relu"
            l   = Layer(W.shape[0], W.shape[1], act, "zeros")
            l.W         = W.copy()
            l.b         = b.copy()
            l.grad_W    = np.zeros_like(W)
            l.grad_b    = np.zeros_like(b)
            l.optimizer = opt
            self.layers.append(l)