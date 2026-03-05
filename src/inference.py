import argparse
import json
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data


def parse_arguments(args_list=None):
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",        type=int,   default=20)
    p.add_argument("-b",   "--batch_size",    type=int,   default=32)
    p.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                   choices=["mean_squared_error", "cross_entropy"])
    p.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0.0005)
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128])
    p.add_argument("-a",   "--activation",    type=str,   nargs="+",
                   default=["relu", "relu"], choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi",  "--weight_init",   type=str,   default="xavier",
                   choices=["random", "xavier", "zeros"])
    p.add_argument("-wp",  "--wandb_project", type=str,   default="da6401_assignment_1")
    p.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    p.add_argument("--model_path",  type=str, default="src/best_model.npy")
    p.add_argument("--config_path", type=str, default="src/best_config.json")
    if args_list is not None:
        return p.parse_args(args_list)
    return p.parse_args()

# alias
parse_args = parse_arguments


def _find_config(model_path, config_path):
    """Try multiple locations to find best_config.json."""
    candidates = [
        config_path,
        os.path.join(os.path.dirname(model_path), "best_config.json"),
        "src/best_config.json",
        "best_config.json",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def load_model(model_path, config_path=None):
    # ── 1. Find config ────────────────────────────────────────────────
    found_cfg = _find_config(model_path, config_path)

    # ── 2. Build args: start from defaults ───────────────────────────
    model_args = parse_arguments([])

    # Apply any explicit CLI flags (e.g. --dataset fashion_mnist from autograder)
    if len(sys.argv) > 1:
        try:
            cli = parse_arguments()
            for k, v in vars(cli).items():
                setattr(model_args, k, v)
        except Exception:
            pass

    # Override with saved config — this controls the architecture
    if found_cfg:
        with open(found_cfg) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(model_args, k, v)
    else:
        print("WARNING: best_config.json not found anywhere — using CLI defaults")

    # ── 3. Build model ────────────────────────────────────────────────
    model = NeuralNetwork(model_args)
    model.args = model_args

    # ── 4. Load weights ───────────────────────────────────────────────
    raw = np.load(model_path, allow_pickle=True)
    if isinstance(raw, np.ndarray):
        weights = raw.item() if raw.ndim == 0 else raw.tolist()
    else:
        weights = raw
    model.set_weights(weights)

    # ── 5. Sanity check ───────────────────────────────────────────────
    dummy = np.zeros((1, 784), dtype=np.float64)
    out   = model.forward(dummy)
    assert out.shape == (1, 10), \
        f"Output shape {out.shape} != (1,10) — weight/config mismatch"

    return model


def run_inference():
    args  = parse_arguments()
    model = load_model(args.model_path, args.config_path)

    dataset = getattr(model.args, "dataset", args.dataset)
    _, _, _, _, x_test, y_test = load_data(dataset)

    logits = model.forward(x_test)
    preds  = np.argmax(logits, axis=1)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test,  preds, average="macro", zero_division=0)
    f1   = f1_score(y_test,  preds, average="macro", zero_division=0)

    print("\nTest Set Evaluation")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return f1


if __name__ == "__main__":
    run_inference()