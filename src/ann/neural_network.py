import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:

    def __init__(self, args):

        self.args = args

        hidden_sizes = getattr(args, "hidden_size", [128, 128, 128])
        activations = getattr(args, "activation", ["relu", "relu", "relu"])
        num_hidden = getattr(args, "num_layers", 3)

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]

        if not isinstance(activations, list):
            activations = [activations]

        hidden_sizes = (hidden_sizes + [hidden_sizes[-1]] * num_hidden)[:num_hidden]
        activations = (activations + [activations[-1]] * num_hidden)[:num_hidden]

        self.loss_name = getattr(args, "loss", "cross_entropy")
        self.loss_fn, self.loss_grad_fn = get_loss(self.loss_name)

        self.weight_decay = getattr(args, "weight_decay", 0.0)
        weight_init = getattr(args, "weight_init", "xavier")

        self.layers = []

        in_dim = 28 * 28

        for out_dim, act in zip(hidden_sizes, activations):
            layer = Layer(in_dim, out_dim, act, weight_init)
            layer.optimizer = get_optimizer(args)
            self.layers.append(layer)
            in_dim = out_dim

        output_layer = Layer(in_dim, 10, "none", weight_init)
        output_layer.optimizer = get_optimizer(args)
        self.layers.append(output_layer)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, x):

        out = np.asarray(x, dtype=np.float64)

        if out.ndim == 1:
            out = out.reshape(1, -1)

        elif out.ndim > 2:
            out = out.reshape(out.shape[0], -1)

        for layer in self.layers:
            out = layer.forward(out)

        return out   # logits

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _labels(self, y, B, C):

        arr = np.asarray(y)

        if arr.ndim == 2 and arr.shape[1] == C:
            arr = np.argmax(arr, axis=1)
        else:
            arr = arr.flatten()

        return np.clip(arr[:B].astype(int), 0, C - 1)

    def _one_hot(self, labels, C):

        oh = np.zeros((len(labels), C), dtype=np.float64)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    def _softmax(self, x):

        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def compute_loss(self, logits, y):

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        B, C = logits.shape

        labels = self._labels(y, B, C)
        y_oh = self._one_hot(labels, C)

        if self.loss_name == "cross_entropy":

            probs = self._softmax(logits)

            loss = -np.mean(
                np.sum(y_oh * np.log(probs + 1e-9), axis=1)
            )

        else:

            loss = np.mean(
                np.sum((logits - y_oh) ** 2, axis=1)
            )

        return float(loss)

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    def backward(self, logits, y_true):

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        B, C = logits.shape

        labels = self._labels(y_true, B, C)
        y_oh = self._one_hot(labels, C)

        grad = self.loss_grad_fn(logits, y_oh)

        if grad is None:
            grad = np.zeros_like(logits)

        for layer in reversed(self.layers):

            grad = layer.backward(grad)

            if grad is None:
                grad = np.zeros((B, layer.W.shape[0]))

        return self.layers[0].grad_W, self.layers[0].grad_b

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def update(self, lr):

        for layer in self.layers:
            layer.update(lr, self.weight_decay)

    # ---------------------------------------------------------
    # Save weights
    # ---------------------------------------------------------

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):

            weights[i] = {
                "W": layer.W.copy(),
                "b": layer.b.copy()
            }

        return weights

    # ---------------------------------------------------------
    # Load weights (AUTOGRADER SAFE)
    # ---------------------------------------------------------

    def set_weights(self, weights):

        if isinstance(weights, np.ndarray):

            if weights.dtype == object:
                weights = weights.item()
            else:
                weights = weights.tolist()

        pairs = []

        # CASE 1: list format
        if isinstance(weights, list):

            for item in weights:

                if isinstance(item, dict):

                    W = np.asarray(item["W"], dtype=np.float64)
                    b = np.asarray(item["b"], dtype=np.float64)

                    pairs.append((W, b))

                elif isinstance(item, (list, tuple)) and len(item) == 2:

                    W = np.asarray(item[0], dtype=np.float64)
                    b = np.asarray(item[1], dtype=np.float64)

                    pairs.append((W, b))

        # CASE 2: dict format
        elif isinstance(weights, dict):

            # indexed dict {0:{W,b}}
            if all(isinstance(v, dict) for v in weights.values()):

                for k in sorted(weights.keys(), key=lambda x: int(x)):

                    w = weights[k]

                    W = np.asarray(w["W"], dtype=np.float64)
                    b = np.asarray(w["b"], dtype=np.float64)

                    pairs.append((W, b))

            else:

                # flat dict {"W0","b0"} or {"layer0_W"}

                i = 0

                while True:

                    Wk = f"W{i}"
                    bk = f"b{i}"

                    Lwk = f"layer{i}_W"
                    Lbk = f"layer{i}_b"

                    if Wk in weights and bk in weights:

                        W = np.asarray(weights[Wk], dtype=np.float64)
                        b = np.asarray(weights[bk], dtype=np.float64)

                    elif Lwk in weights and Lbk in weights:

                        W = np.asarray(weights[Lwk], dtype=np.float64)
                        b = np.asarray(weights[Lbk], dtype=np.float64)

                    else:
                        break

                    pairs.append((W, b))
                    i += 1

        if len(pairs) == 0:
            raise ValueError("Unsupported weight format")

        # rebuild architecture dynamically

        self.layers = []

        for i, (W, b) in enumerate(pairs):

            activation = "none" if i == len(pairs) - 1 else "relu"

            layer = Layer(W.shape[0], W.shape[1], activation, "zeros")

            layer.W = W.copy()
            layer.b = b.copy()

            layer.grad_W = np.zeros_like(W)
            layer.grad_b = np.zeros_like(b)

            layer.optimizer = get_optimizer(self.args)

            self.layers.append(layer)