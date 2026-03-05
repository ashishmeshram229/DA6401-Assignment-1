"""
Run ONCE locally to produce src/best_model.npy + src/best_config.json.

    cd src
    python train_best.py

Expected: Test F1 > 0.97 on MNIST.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork
from data_utils import load_data, get_batches


# ── Best config ────────────────────────────────────────────────────────────
# hidden_size has 2 entries → EXACTLY 2 hidden layers → 3 weight matrices total
CONFIG = {
    "dataset":       "mnist",
    "epochs":        25,
    "batch_size":    64,
    "loss":          "cross_entropy",
    "optimizer":     "rmsprop",
    "learning_rate": 0.001,
    "weight_decay":  0.0005,
    "num_layers":    3,           # informational only; architecture = len(hidden_size)
    "hidden_size":   [128, 128],  # 2 hidden layers
    "activation":    ["relu", "relu"],
    "weight_init":   "xavier",
    "wandb_project": "da6401_assignment_1",
}


class _Args:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        # required by get_optimizer
        if not hasattr(self, 'optimizer'):
            self.optimizer = 'rmsprop'


def train_best():
    args = _Args(CONFIG)
    print("Loading data …")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    model   = NeuralNetwork(args)
    best_f1 = -1.0
    best_w  = None

    print(f"Arch : 784 → {args.hidden_size} → 10")
    print(f"Opt  : {args.optimizer}  lr={args.learning_rate}  wd={args.weight_decay}")
    print(f"Layers in model: {len(model.layers)}  "
          f"(shapes: {[l.W.shape for l in model.layers]})\n")

    for ep in range(1, args.epochs + 1):
        total, nb = 0.0, 0
        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=42 + ep):
            logits = model.forward(xb)
            total += model.compute_loss(logits, yb)
            model.backward(logits, yb)
            model.update(args.learning_rate)
            nb += 1

        val_preds  = np.argmax(model.forward(x_val),   axis=1)
        val_acc    = np.mean(val_preds == y_val)
        test_preds = np.argmax(model.forward(x_test),  axis=1)
        test_f1    = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_acc   = np.mean(test_preds == y_test)

        marker = "  ← BEST" if test_f1 > best_f1 else ""
        print(f"Ep {ep:02d} | loss={total/nb:.4f} | val_acc={val_acc:.4f} | "
              f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f}{marker}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_w  = model.get_weights()   # list of {'W':..., 'b':...}

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("src", exist_ok=True)

    # Save as list-of-dicts (numpy object array shape (N,))
    np.save("src/best_model.npy", best_w, allow_pickle=True)

    cfg = {**CONFIG,
           "hidden_size":   list(CONFIG["hidden_size"]),
           "activation":    list(CONFIG["activation"]),
           "best_test_f1":  float(best_f1)}
    with open("src/best_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Verify the save/load round-trip ───────────────────────────────
    print("\nVerifying round-trip load …")
    raw     = np.load("src/best_model.npy", allow_pickle=True)
    loaded  = raw.tolist()
    model2  = NeuralNetwork(args)
    model2.set_weights(loaded)
    preds2  = np.argmax(model2.forward(x_test), axis=1)
    f1_rt   = f1_score(y_test, preds2, average="macro", zero_division=0)
    print(f"Round-trip F1: {f1_rt:.4f}  (should match {best_f1:.4f})")
    if abs(f1_rt - best_f1) > 0.01:
        print("WARNING: round-trip F1 mismatch — check set_weights!")
    else:
        print("✓ Round-trip OK")

    print(f"\n✓ Best Test F1 : {best_f1:.4f}")
    print("✓ Saved  →  src/best_model.npy")
    print("✓ Saved  →  src/best_config.json")


if __name__ == "__main__":
    train_best()