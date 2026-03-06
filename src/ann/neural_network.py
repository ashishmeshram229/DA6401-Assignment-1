import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.layer        import Layer
from ann.optimizers          import optimiser
from ann.losses import loss_func, loss_gradient
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)


class NeuralNetwork: # A simple feedforward neural network with configurable architecture, activations, and optimizers

    def __init__(self, cli_args):


        self.args   = cli_args
        self.layers = []
        self._build()
        self.optimizer = optimiser[cli_args.optimizer](
            lr           = cli_args.learning_rate,
            weight_decay = cli_args.weight_decay,
        )

        self.optimizer.init_state(self.layers)



    def _build(self): # Build the network architecture based on CLI arguments
        a           = self.args
        num_layers  = getattr(a, 'num_layers',  3)


        hidden_size = getattr(a, 'hidden_size', [128] * (num_layers - 1))
        activation  = getattr(a, 'activation',  'relu')
        weight_init = getattr(a, 'weight_init', 'xavier')
        # allow override of input/output dims for autograder dummy models
        input_dim   = getattr(a, 'input_dim',  784)
        output_dim  = getattr(a, 'output_dim', 10)


        # Always normalise activation to a single string

        if isinstance(activation, list):
            activation = activation[0]




        # Always normalise hidden_size to a list
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size] * (num_layers - 1)

        num_hidden = num_layers - 1
        if len(hidden_size) < num_hidden: # Pad with last value to match num_layers - 1
            
            hidden_size = hidden_size + [hidden_size[-1]] * (num_hidden - len(hidden_size))


        elif len(hidden_size) > num_hidden:
            hidden_size = hidden_size[:num_hidden] # Trim to match num_layers - 1

        dims = [input_dim] + hidden_size + [output_dim]



        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None
            self.layers.append(Layer(dims[i], dims[i + 1], act, weight_init))

    def forward(self, X): # Forward pass through the network

        out = np.asarray(X, dtype=np.float64)

        for layer in self.layers:
            out = layer.forward(out)


        return out  # raw logits


    def backward(self, y_true, y_pred): # Backward pass: compute gradients for all layers based on loss gradient at output 

        loss_key = getattr(self.args, 'loss', 'cross_entropy')
        delta    = loss_gradient[loss_key](y_pred, y_true)
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

        for epoch in range(epochs): # Shuffle training data at the start of each epoch

            idx              = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]
            epoch_loss       = 0.0
            correct          = 0  # track correct predictions across batches — avoids re-running full forward pass after epoch

            for start in range(0, n, batch_size):


                Xb          = X_train[start:start + batch_size]
                yb          = y_train[start:start + batch_size]
                logits       = self.forward(Xb)

                epoch_loss  += loss_func[loss_key](logits, yb) * len(yb)
                correct     += np.sum(np.argmax(logits, axis=1) == yb)  # accumulate correct predictions

                self.backward(yb, logits)
                self.update_weights()


            epoch_loss /= n
            train_acc   = correct / n  # computed from batch logits — no extra forward pass needed

            log = {"epoch":      epoch + 1,
                   "train_loss": epoch_loss,
                   "train_acc":  train_acc}


            if X_val is not None: # Evaluate on validation set and log metrics (also used for model selection based on best F1 score)

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
                  f"| train_acc: {train_acc:.4f}", end="")
            if X_val is not None:
                print(f" | val_acc: {log['val_acc']:.4f}", end="")
            print()

        return best_weights

    def evaluate(self, X, y): # Compute loss and metrics on given dataset (used for validation and test evaluation)

        loss_key = getattr(self.args, 'loss', 'cross_entropy')
        logits   = self.forward(X)
        loss     = loss_func[loss_key](logits, y)
        preds    = np.argmax(logits, axis=1)
        return {
            "loss":      loss,
            "accuracy":  accuracy_score(y, preds),
            "f1":        f1_score(y,    preds, average="macro", zero_division=0),
            "precision": precision_score(y, preds, average="macro", zero_division=0),
            "recall":    recall_score(y,  preds, average="macro", zero_division=0),
            "logits":    logits,
        }


    def get_weights(self): # Return a dictionary of all layer weights and biases

        weights = {}
        for i, l in enumerate(self.layers):
            weights[f"W{i}"] = l.W.copy()
            weights[f"b{i}"] = l.b.copy()

        return weights

    def set_weights(self, weights): # Set layer weights and biases from a given dictionary

        if isinstance(weights, np.ndarray):
            weights = weights.item() if weights.ndim == 0 else list(weights)

        # Parse into list of (W, b) pairs
        pairs = []
        if isinstance(weights, (tuple, list)):
            for item in weights:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    pairs.append((np.asarray(item[0], dtype=np.float64),
                                  np.asarray(item[1], dtype=np.float64)))
                elif isinstance(item, dict):
                    pairs.append((np.asarray(item.get("W", item.get("w")), dtype=np.float64),
                                  np.asarray(item["b"], dtype=np.float64)))
        elif isinstance(weights, dict):
            n = len([k for k in weights if k.startswith("W")])
            for i in range(n):
                if f"W{i}" in weights:
                    pairs.append((np.asarray(weights[f"W{i}"], dtype=np.float64),
                                  np.asarray(weights[f"b{i}"], dtype=np.float64)))
                elif str(i) in weights:
                    pairs.append((np.asarray(weights[str(i)]["W"], dtype=np.float64),
                                  np.asarray(weights[str(i)]["b"], dtype=np.float64)))
        else:
            print(f"WARNING: set_weights could not parse {type(weights)}")
            return

        if not pairs:
            print("WARNING: set_weights found no weight pairs")
            return

        # Rebuild layers if architecture doesn't match loaded weights
        if len(pairs) != len(self.layers) or any(
            pairs[i][0].shape != self.layers[i].W.shape for i in range(len(pairs))
        ):
            activation  = getattr(self.args, 'activation', 'relu')
            if isinstance(activation, list):
                activation = activation[0]
            weight_init = getattr(self.args, 'weight_init', 'xavier')

            self.layers = []
            for i, (W, b) in enumerate(pairs):
                act = activation if i < len(pairs) - 1 else None
                layer = Layer(W.shape[0], W.shape[1], act, weight_init)
                layer.W = W.copy()
                layer.b = b.reshape(1, -1).copy()
                self.layers.append(layer)
            self.optimizer.init_state(self.layers) # re-initialise optimiser state for new architecture
            return

        # Shapes match — just copy values
        for i, (W, b) in enumerate(pairs):
            self.layers[i].W = W.copy()
            self.layers[i].b = b.reshape(1, -1).copy()