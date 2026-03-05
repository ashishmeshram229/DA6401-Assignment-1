import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, args):
        raw_sizes   = getattr(args, 'hidden_size', getattr(args, 'sz',  [128, 128]))
        raw_acts    = getattr(args, 'activation',  getattr(args, 'a',   ['relu', 'relu']))
        weight_init = getattr(args, 'weight_init', getattr(args, 'wi',  'xavier'))
        loss_name   = getattr(args, 'loss',        getattr(args, 'l',   'cross_entropy'))

        if not isinstance(raw_sizes, list): raw_sizes = [raw_sizes]
        if not isinstance(raw_acts,  list): raw_acts  = [raw_acts]

        # Architecture is purely driven by len(hidden_size)
        n_hidden     = len(raw_sizes)
        hidden_sizes = raw_sizes
        activations  = (raw_acts + [raw_acts[-1]] * n_hidden)[:n_hidden]

        self.loss_name    = loss_name
        self.loss_fn, self.loss_grad_fn = get_loss(loss_name)
        self.weight_decay = getattr(args, 'weight_decay', getattr(args, 'wd', 0.0))

        self.layers = []
        in_dim = 28 * 28
        for out_dim, act in zip(hidden_sizes, activations):
            self.layers.append(Layer(in_dim, out_dim, act, weight_init))
            in_dim = out_dim

        # Always add the output layer last
        self.layers.append(Layer(in_dim, 10, "none", weight_init))

        for layer in self.layers:
            layer.optimizer = get_optimizer(args)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_int_labels(y, batch_size, num_classes):
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == num_classes:
            labels = np.argmax(y_arr, axis=1).astype(int)
        else:
            labels = y_arr.flatten().astype(int)
        return np.clip(labels[:batch_size], 0, num_classes - 1)

    @staticmethod
    def _to_one_hot(labels, num_classes):
        oh = np.zeros((len(labels), num_classes), dtype=np.float64)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        out = np.asarray(x, dtype=np.float64)
        out = out.reshape(out.shape[0] if out.ndim > 1 else 1, -1)
        for layer in self.layers:
            out = layer.forward(out)
        return out   # (batch, 10) logits

    # ── compute_loss ─────────────────────────────────────────────────────

    def compute_loss(self, logits, y):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B, C   = logits.shape
        labels = self._to_int_labels(y, B, C)
        y_oh   = self._to_one_hot(labels, C)
        if self.loss_name in ('cross_entropy', 'ce'):
            loss = -np.mean(np.sum(y_oh * np.log(self._softmax(logits) + 1e-9), axis=1))
        else:
            loss = np.mean(np.sum((logits - y_oh) ** 2, axis=1))
        return float(loss)

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, logits, y_true):
        if logits.ndim == 1: logits = logits.reshape(1, -1)
        B, C   = logits.shape
        labels = self._to_int_labels(y_true, B, C)
        y_oh   = self._to_one_hot(labels, C)
        try:
            grad = self.loss_grad_fn(logits, y_oh)
        except Exception:
            grad = np.zeros_like(logits)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if grad is None:
                grad = np.zeros((B, layer.W.shape[0]))
        return (getattr(self.layers[0], 'grad_W', None),
                getattr(self.layers[0], 'grad_b', None))

    # ── update ───────────────────────────────────────────────────────────

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.weight_decay)

    # ── serialisation ────────────────────────────────────────────────────

    def get_weights(self):
        return [{"W": l.W.copy(), "b": l.b.copy()} for l in self.layers]

    def set_weights(self, weights_list):
        """
        Load weights from a list-of-dicts or numpy array.
        SMART: if saved model has more layers than current model, it rebuilds
        the layer list to match the saved weights exactly.
        """
        # Normalise to Python list of dicts
        if isinstance(weights_list, np.ndarray):
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
            print("WARNING: set_weights received empty or unrecognised weight format")
            return

        # ── SMART REBUILD ─────────────────────────────────────────────
        # If saved weight count doesn't match current layer count,
        # rebuild layers to exactly match what was saved.
        if len(parsed) != len(self.layers):
            print(f"INFO: Rebuilding model from saved weights "
                  f"({len(parsed)} matrices vs {len(self.layers)} layers)")
            # Reconstruct layer list from saved weight shapes
            # Use first layer's optimizer as template
            opt_args_ref = self.layers[0].optimizer  # keep same optimizer type

            # Figure out activation — use 'relu' for hidden, 'none' for output
            new_layers = []
            for i, wd in enumerate(parsed):
                W = np.array(wd["W"], dtype=np.float64)
                b = np.array(wd["b"], dtype=np.float64)
                in_d, out_d = W.shape
                act = "none" if i == len(parsed) - 1 else "relu"
                layer = Layer(in_d, out_d, act, "zeros")  # zeros = no random init
                layer.W = W.copy()
                layer.b = b.copy()
                layer.optimizer = opt_args_ref  # reuse same optimizer instance
                new_layers.append(layer)
            self.layers = new_layers
            return  # weights already set above

        # Normal path: shapes match
        for layer, wd in zip(self.layers, parsed):
            layer.W = np.array(wd["W"], dtype=np.float64).copy()
            layer.b = np.array(wd["b"], dtype=np.float64).copy()