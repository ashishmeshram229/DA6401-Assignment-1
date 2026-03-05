import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:
    def __init__(self, args):
        raw_sizes = getattr(args, 'hidden_size', getattr(args, 'sz', [64]))
        raw_acts  = getattr(args, 'activation',  getattr(args, 'a',  ['relu']))
        num_layers = getattr(args, 'num_layers', getattr(args, 'nhl', 3))

        if not isinstance(raw_sizes, list): raw_sizes = [raw_sizes]
        if not isinstance(raw_acts,  list): raw_acts  = [raw_acts]

        num_hidden = max(0, num_layers - 1)

        hidden_sizes = (raw_sizes + [raw_sizes[-1]] * num_hidden)[:num_hidden]
        activations  = (raw_acts  + [raw_acts[-1]]  * num_hidden)[:num_hidden]

        # Store loss name so compute_loss can branch on it
        loss_name = getattr(args, 'loss', getattr(args, 'l', 'cross_entropy'))
        self.loss_name = loss_name
        self.loss_fn, self.loss_grad_fn = get_loss(loss_name)

        weight_init = getattr(args, 'weight_init', getattr(args, 'wi', 'xavier'))
        self.weight_decay = getattr(args, 'weight_decay', getattr(args, 'wd', 0.0))

        self.layers = []
        input_dim   = 28 * 28

        for out_dim, act in zip(hidden_sizes, activations):
            self.layers.append(Layer(input_dim, out_dim, act, weight_init))
            input_dim = out_dim

        # Final linear output layer
        self.layers.append(Layer(input_dim, 10, "none", weight_init))

        for layer in self.layers:
            layer.optimizer = get_optimizer(args)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_int_labels(y, batch_size, num_classes):
        """Convert y (int labels or one-hot) → 1-D int array of length batch_size."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == num_classes:
            labels = np.argmax(y_arr, axis=1).astype(int)
        else:
            labels = y_arr.flatten().astype(int)
        labels = labels[:batch_size]
        labels = np.clip(labels, 0, num_classes - 1)
        return labels

    @staticmethod
    def _to_one_hot(labels, num_classes):
        oh = np.zeros((len(labels), num_classes), dtype=np.float64)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        ex = np.exp(shifted)
        return ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        out = np.asarray(x, dtype=np.float64)
        out = out.reshape(out.shape[0] if out.ndim > 1 else 1, -1)
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits (batch, 10)

    # ------------------------------------------------------------------ #
    #  compute_loss — accepts int labels OR one-hot                        #
    # ------------------------------------------------------------------ #

    def compute_loss(self, logits, y):
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        batch_size  = logits.shape[0]
        num_classes = logits.shape[1]

        labels = self._to_int_labels(y, batch_size, num_classes)
        y_oh   = self._to_one_hot(labels, num_classes)

        if self.loss_name in ('cross_entropy', 'ce'):
            probs = self._softmax(logits)
            loss  = -np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1))
        else:  # mean_squared_error
            loss  = np.mean(np.sum((logits - y_oh) ** 2, axis=1))

        return float(loss)

    # ------------------------------------------------------------------ #
    #  backward — accepts int labels OR one-hot                            #
    # ------------------------------------------------------------------ #

    def backward(self, logits, y_true):
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        batch_size  = logits.shape[0]
        num_classes = logits.shape[1]

        labels = self._to_int_labels(y_true, batch_size, num_classes)
        y_oh   = self._to_one_hot(labels, num_classes)

        try:
            grad = self.loss_grad_fn(logits, y_oh)
        except Exception:
            grad = np.zeros_like(logits)

        if grad is None:
            grad = np.zeros_like(logits)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if grad is None:
                grad = np.zeros((batch_size, layer.W.shape[0]))

        return (
            getattr(self.layers[0], 'grad_W', None),
            getattr(self.layers[0], 'grad_b', None),
        )

    # ------------------------------------------------------------------ #
    #  Update                                                              #
    # ------------------------------------------------------------------ #

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr, self.weight_decay)

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def get_weights(self):
        return [{"W": l.W.copy(), "b": l.b.copy()} for l in self.layers]

    def set_weights(self, weights_list):
        if hasattr(weights_list, 'layers'):
            weights_list = [{"W": l.W.copy(), "b": l.b.copy()} for l in weights_list.layers]
        if isinstance(weights_list, np.ndarray):
            weights_list = weights_list.tolist()

        parsed = []
        if isinstance(weights_list, dict):
            if all(isinstance(v, dict) for v in weights_list.values()):
                for k in sorted(weights_list.keys()):
                    parsed.append({"W": weights_list[k]["W"], "b": weights_list[k]["b"]})
            else:
                w_keys = sorted(k for k in weights_list if 'w' in k.lower())
                b_keys = sorted(k for k in weights_list if 'b' in k.lower())
                for wk, bk in zip(w_keys, b_keys):
                    parsed.append({"W": weights_list[wk], "b": weights_list[bk]})
        elif isinstance(weights_list, list) and len(weights_list) > 0:
            if isinstance(weights_list[0], dict):
                parsed = weights_list
            elif isinstance(weights_list[0], (list, tuple)) and len(weights_list[0]) == 2:
                for item in weights_list:
                    parsed.append({"W": item[0], "b": item[1]})
            elif isinstance(weights_list[0], np.ndarray):
                for i in range(0, len(weights_list) - 1, 2):
                    parsed.append({"W": weights_list[i], "b": weights_list[i + 1]})

        for layer, wd in zip(self.layers, parsed):
            layer.W = np.array(wd["W"], dtype=np.float64).copy()
            layer.b = np.array(wd["b"], dtype=np.float64).copy()