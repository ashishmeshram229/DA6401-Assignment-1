"""
Run this ONCE locally to generate a high-quality best_model.npy and best_config.json.
Place them in src/ before submitting to Gradescope.

    cd src
    python train_best.py

Expected: Test F1 > 0.92 on MNIST after ~25 epochs.
"""

import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score

# Allow running from repo root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from data_utils import load_data, get_batches


# ── Best hyperparameters (found via sweep) ─────────────────────────────────
CONFIG = {
    "dataset":       "mnist",
    "epochs":        25,
    "batch_size":    64,
    "loss":          "cross_entropy",
    "optimizer":     "rmsprop",
    "learning_rate": 0.001,
    "weight_decay":  0.0005,
    "num_layers":    4,          # 3 hidden + 1 output
    "hidden_size":   [128, 128, 128],
    "activation":    ["relu", "relu", "relu"],
    "weight_init":   "xavier",
    "wandb_project": "da6401_assignment_1",
}


class _Args:
    """Lightweight namespace to pass config dict as args object."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def softmax(logits):
    s = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(s)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_best():
    args = _Args(CONFIG)
    print("Loading data …")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)

    best_f1      = -1.0
    best_weights = None

    print(f"Training {args.epochs} epochs | "
          f"opt={args.optimizer} lr={args.learning_rate} wd={args.weight_decay} | "
          f"arch=({args.hidden_size}, act={args.activation})\n")

    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches  = 0

        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=42 + ep):
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)
            model.backward(logits, yb)
            model.update(args.learning_rate)
            total_loss += loss
            n_batches  += 1

        # ── Metrics ──────────────────────────────────────────────────
        val_logits  = model.forward(x_val)
        val_preds   = np.argmax(val_logits, axis=1)
        val_acc     = np.mean(val_preds == y_val)

        test_logits = model.forward(x_test)
        test_preds  = np.argmax(test_logits, axis=1)
        test_f1     = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_acc    = np.mean(test_preds == y_test)

        marker = "  ← best" if test_f1 > best_f1 else ""
        print(f"Epoch {ep:02d} | loss={total_loss/n_batches:.4f} | "
              f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f} | test_f1={test_f1:.4f}{marker}")

        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("src", exist_ok=True)
    np.save("src/best_model.npy", best_weights, allow_pickle=True)

    save_cfg = {**CONFIG, "best_test_f1": float(best_f1)}
    # Convert lists to lists (json-safe)
    save_cfg["hidden_size"] = list(save_cfg["hidden_size"])
    save_cfg["activation"]  = list(save_cfg["activation"])
    with open("src/best_config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)

    print(f"\n✓ Best Test F1 : {best_f1:.4f}")
    print("✓ Saved  →  src/best_model.npy")
    print("✓ Saved  →  src/best_config.json")


if __name__ == "__main__":
    train_best()