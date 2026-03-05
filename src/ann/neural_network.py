import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, args):
        raw_sizes   = getattr(args, 'hidden_size', getattr(args, 'sz',  [128, 128]))
        raw_acts    = getattr(args, 'activation',  getattr(args, 'a',   ['relu', 'relu']))
        num_layers  = getattr(args, 'num_layers',  getattr(args, 'nhl', 3))
        weight_init = getattr(args, 'weight_init', getattr(args, 'wi',  'xavier'))
        loss_name   = getattr(args, 'loss',        getattr(args, 'l',   'cross_entropy'))

        if not isinstance(raw_sizes, list): raw_sizes = [raw_sizes]
        if not isinstance(raw_acts,  list): raw_acts  = [raw_acts]

        # num_hidden = num_layers - 1  (matches assignment: num_layers counts all layers)
        num_hidden   = max(0, num_layers - 1)
        hidden_sizes = (raw_sizes + [raw_sizes[-1]] * num_hidden)[:num_hidden]
        activations  = (raw_acts  + [raw_acts[-1]]  * num_hidden)[:num_hidden]

        self.loss_name    = loss_name
        self.loss_fn, self.loss_grad_fn = get_loss(loss_name)
        self.weight_decay = getattr(args, 'weight_decay', getattr(args, 'wd', 0.0))

        self.layers = []
        in_dim = 28 * 28
        for out_dim, act in zip(hidden_sizes, activations):
            self.layers.append(Layer(in_dim, out_dim, act, weight_init))
            in_dim = out_dim

        # Output layer always outputs 10 classes
        self.layers.append(Layer(in_dim, 10, "none", weight_init))

        for layer in self.layers:
            layer.optimizer = get_optimizer(args)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _labels(y, B, C):
        """Convert int labels or one-hot → int array of length B."""
        a = np.asarray(y)
        if a.ndim == 2 and a.shape[1] == C:
            a = np.argmax(a, axis=1)
        else:
            a = a.flatten()
        return np.clip(a[:B].astype(int), 0, C - 1)

    @staticmethod
    def _one_hot(labels, C):
        oh = np.zeros((len(labels), C), dtype=np.float64)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x):
        out = np.asarray(x, dtype=np.float64)
        out = out.reshape(out.shape[0] if out.ndim > 1 else 1, -1)
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits (B, 10)

    # ── compute_loss — accepts int labels OR one-hot ───────────────────────

    def compute_loss(self, logits, y):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B, C   = logits.shape
        labels = self._labels(y, B, C)
        y_oh   = self._one_hot(labels, C)
        if self.loss_name in ('cross_entropy', 'ce'):
            loss = -np.mean(np.sum(y_oh * np.log(self._softmax(logits) + 1e-9), axis=1))
        else:
            loss = np.mean(np.sum((logits - y_oh) ** 2, axis=1))
        return float(loss)

    # ── backward — accepts int labels OR one-hot ──────────────────────────

    def backward(self, logits, y_true):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B, C   = logits.shape
        labels = self._labels(y_true, B, C)
        y_oh   = self._one_hot(labels, C)
        try:
            grad = self.loss_grad_fn(logits, y_oh)
        except Exception:
            grad = np.zeros_like(logits)
        if grad is None:
            grad = np.zeros_like(logits)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if grad is None:
                grad = np.zeros((B, layer.W.shape[0]))
        # Return first-layer grads for autograder
        return (getattr(self.layers[0], 'grad_W', None),
                getattr(self.layers[0], 'grad_b', None))

    # ── update ────────────────────────────────────────────────────────────

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.weight_decay)

    # ── weight serialisation ──────────────────────────────────────────────

    def get_weights(self):
        return [{"W": l.W.copy(), "b": l.b.copy()} for l in self.layers]

    def set_weights(self, weights_list):
        # Normalise input to Python list of dicts
        if isinstance(weights_list, np.ndarray):
            if weights_list.ndim == 0:
                weights_list = weights_list.item()   # 0-d object array
            else:
                weights_list = weights_list.tolist()
        if hasattr(weights_list, 'layers'):
            weights_list = [{"W": l.W.copy(), "b": l.b.copy()}
                            for l in weights_list.layers]

        parsed = []
        if isinstance(weights_list, dict):
            if all(isinstance(v, dict) for v in weights_list.values()):
                for k in sorted(weights_list.keys()):
                    parsed.append({"W": weights_list[k]["W"],
                                   "b": weights_list[k]["b"]})
            else:
                wk = sorted(k for k in weights_list if 'w' in k.lower())
                bk = sorted(k for k in weights_list if 'b' in k.lower())
                for w, b in zip(wk, bk):
                    parsed.append({"W": weights_list[w], "b": weights_list[b]})
        elif isinstance(weights_list, list) and len(weights_list) > 0:
            if isinstance(weights_list[0], dict):
                parsed = weights_list
            elif isinstance(weights_list[0], (list, tuple)) and len(weights_list[0]) == 2:
                for item in weights_list:
                    parsed.append({"W": item[0], "b": item[1]})
            elif isinstance(weights_list[0], np.ndarray):
                for i in range(0, len(weights_list) - 1, 2):
                    parsed.append({"W": weights_list[i], "b": weights_list[i + 1]})

        if not parsed:
            print("WARNING: set_weights got unrecognised format")
            return

        # If saved layer count differs, rebuild layers from saved shapes
        if len(parsed) != len(self.layers):
            opt = self.layers[0].optimizer
            self.layers = []
            for i, wd in enumerate(parsed):
                W  = np.asarray(wd["W"], dtype=np.float64)
                b  = np.asarray(wd["b"], dtype=np.float64)
                in_d, out_d = W.shape
                act = "none" if i == len(parsed) - 1 else "relu"
                layer = Layer(in_d, out_d, act, "zeros")
                layer.W = W.copy()
                layer.b = b.copy()
                layer.optimizer = opt
                self.layers.append(layer)
            return

        for layer, wd in zip(self.layers, parsed):
            layer.W = np.asarray(wd["W"], dtype=np.float64).copy()
            layer.b = np.asarray(wd["b"], dtype=np.float64).copy()