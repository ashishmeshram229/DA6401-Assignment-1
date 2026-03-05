"""
Neural Network — loss functions embedded directly, zero external file dependencies.
"""
import os, sys
# Guarantee src/ is always on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.neural_layer import Layer
from ann.activations  import ACT_FN, ACT_GRAD, softmax
from ann.optimizers   import OPTIMIZERS
from sklearn.metrics  import (accuracy_score, f1_score,
                              precision_score, recall_score)

# ── Loss functions embedded here so objective_functions.py is optional ────────

def _cross_entropy(logits, y_true):
    probs = softmax(logits)
    n     = y_true.shape[0]
    return float(np.mean(-np.log(probs[np.arange(n), y_true.astype(int)] + 1e-9)))

def _cross_entropy_grad(logits, y_true):
    probs = softmax(logits)
    n     = y_true.shape[0]
    probs[np.arange(n), y_true.astype(int)] -= 1.0
    return probs / n

def _mse(logits, y_true):
    probs   = softmax(logits)
    n, c    = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true.astype(int)] = 1.0
    return float(np.mean((probs - one_hot) ** 2))

def _mse_grad(logits, y_true):
    probs   = softmax(logits)
    n, c    = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true.astype(int)] = 1.0
    diff    = probs - one_hot
    grad    = np.zeros_like(probs)
    for k in range(c):
        dsm        = probs * (np.eye(c)[k] - probs[:, k:k+1])
        grad[:, k] = np.sum((2.0 / c) * diff * dsm, axis=1)
    return grad / n

LOSS_FN = {
    "cross_entropy":      _cross_entropy,
    "mse":                _mse,
    "mean_squared_error": _mse,
}
LOSS_GRAD = {
    "cross_entropy":      _cross_entropy_grad,
    "mse":                _mse_grad,
    "mean_squared_error": _mse_grad,
}

# ── NeuralNetwork class ───────────────────────────────────────────────────────

class NeuralNetwork:

    def __init__(self, cli_args):
        self.args   = cli_args
        self.layers = []
        self._build()
        self.optimizer = OPTIMIZERS[cli_args.optimizer](
            lr           = cli_args.learning_rate,
            weight_decay = cli_args.weight_decay,
        )
        self.optimizer.init_state(self.layers)

    def _build(self):
        a           = self.args
        num_layers  = getattr(a, 'num_layers',  3)
        hidden_size = getattr(a, 'hidden_size', [128] * (num_layers - 1))
        activation  = getattr(a, 'activation',  'relu')
        weight_init = getattr(a, 'weight_init', 'xavier')

        # Always normalise activation to a single string
        if isinstance(activation, list):
            activation = activation[0]

        # Always normalise hidden_size to a list
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size] * (num_layers - 1)

        num_hidden = num_layers - 1
        if len(hidden_size) < num_hidden:
            hidden_size = hidden_size + [hidden_size[-1]] * (num_hidden - len(hidden_size))
        elif len(hidden_size) > num_hidden:
            hidden_size = hidden_size[:num_hidden]

        dims = [784] + hidden_size + [10]
        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None
            self.layers.append(Layer(dims[i], dims[i + 1], act, weight_init))

    def forward(self, X):
        out = np.asarray(X, dtype=np.float64)
        for layer in self.layers:
            out = layer.forward(out)
        return out  # raw logits

    def backward(self, y_true, y_pred):
        loss_key = getattr(self.args, 'loss', 'cross_entropy')
        delta    = LOSS_GRAD[loss_key](y_pred, y_true)
        grads_w, grads_b = [], []
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grads_w.insert(0, layer.grad_W)
            grads_b.insert(0, layer.grad_b)
        return grads_w, grads_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, epochs, batch_size,
              X_val=None, y_val=None, wandb_run=None):
        n        = X_train.shape[0]
        loss_key = getattr(self.args, 'loss', 'cross_entropy')
        best_f1, best_weights = -1, None

        for epoch in range(epochs):
            idx              = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]
            epoch_loss       = 0.0

            for start in range(0, n, batch_size):
                Xb          = X_train[start:start + batch_size]
                yb          = y_train[start:start + batch_size]
                logits       = self.forward(Xb)
                epoch_loss  += LOSS_FN[loss_key](logits, yb) * len(yb)
                self.backward(yb, logits)
                self.update_weights()

            epoch_loss    /= n
            train_metrics  = self.evaluate(X_train, y_train)
            log = {"epoch":      epoch + 1,
                   "train_loss": epoch_loss,
                   "train_acc":  train_metrics["accuracy"]}

            if X_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                log.update({"val_loss": val_metrics["loss"],
                             "val_acc":  val_metrics["accuracy"],
                             "val_f1":   val_metrics["f1"]})
                if val_metrics["f1"] > best_f1:
                    best_f1      = val_metrics["f1"]
                    best_weights = self.get_weights()

            if wandb_run is not None:
                wandb_run.log(log)

            print(f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} "
                  f"| train_acc: {train_metrics['accuracy']:.4f}", end="")
            if X_val is not None:
                print(f" | val_acc: {log['val_acc']:.4f}", end="")
            print()

        return best_weights

    def evaluate(self, X, y):
        loss_key = getattr(self.args, 'loss', 'cross_entropy')
        logits   = self.forward(X)
        loss     = LOSS_FN[loss_key](logits, y)
        preds    = np.argmax(logits, axis=1)
        return {
            "loss":      loss,
            "accuracy":  accuracy_score(y, preds),
            "f1":        f1_score(y,    preds, average="macro", zero_division=0),
            "precision": precision_score(y, preds, average="macro", zero_division=0),
            "recall":    recall_score(y,  preds, average="macro", zero_division=0),
            "logits":    logits,
        }

    def get_weights(self):
        weights = {}
        for i, l in enumerate(self.layers):
            weights[f"W{i}"] = l.W.copy()
            weights[f"b{i}"] = l.b.copy()
        return weights

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray):
            weights = weights.item() if weights.ndim == 0 else list(weights)

        if isinstance(weights, (tuple, list)):
            for i, layer in enumerate(self.layers):
                if i >= len(weights): break
                item = weights[i]
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    layer.W = np.asarray(item[0], dtype=np.float64).copy()
                    layer.b = np.asarray(item[1], dtype=np.float64).copy()
                elif isinstance(item, dict):
                    layer.W = np.asarray(item.get("W", item.get("w")), dtype=np.float64).copy()
                    layer.b = np.asarray(item["b"], dtype=np.float64).copy()
            return

        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f"W{i}" in weights:
                    layer.W = np.asarray(weights[f"W{i}"], dtype=np.float64).copy()
                    layer.b = np.asarray(weights[f"b{i}"], dtype=np.float64).copy()
                elif str(i) in weights:
                    layer.W = np.asarray(weights[str(i)]["W"], dtype=np.float64).copy()
                    layer.b = np.asarray(weights[str(i)]["b"], dtype=np.float64).copy()
                elif i in weights:
                    layer.W = np.asarray(weights[i]["W"], dtype=np.float64).copy()
                    layer.b = np.asarray(weights[i]["b"], dtype=np.float64).copy()
                else:
                    raise KeyError(f"Layer {i} not found. Keys: {list(weights.keys())}")
            return

        print(f"WARNING: set_weights could not parse {type(weights)}")