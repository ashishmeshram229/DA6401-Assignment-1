import argparse
import json
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data


def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0005)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",
                        default=[128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   nargs="+",
                        default=["relu", "relu"],
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi",  "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-wp",  "--wandb_project", type=str,   default="da6401_assignment_1")
    parser.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    parser.add_argument("--model_path",  type=str, default="src/best_model.npy")
    parser.add_argument("--config_path", type=str, default="src/best_config.json")
    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()

# Alias expected by some callers
parse_args = parse_arguments


def load_model(model_path, config_path=None):
    """
    Load a saved NeuralNetwork from disk.
    Architecture is driven entirely by the saved config (hidden_size list).
    """
    # ── 1. Resolve config path ────────────────────────────────────────
    if config_path is None or not os.path.exists(config_path):
        # Try alongside the model file
        config_path = model_path.replace(".npy", ".json")
    if not os.path.exists(config_path):
        # Try the canonical name next to the model
        config_path = os.path.join(os.path.dirname(model_path), "best_config.json")

    # ── 2. Build args from config ─────────────────────────────────────
    # Start from default CLI args (all standard keys present)
    model_args = parse_arguments([])

    # Layer any explicit CLI flags on top (e.g. --dataset fashion_mnist)
    if len(sys.argv) > 1:
        try:
            cli = parse_arguments()
            for k, v in vars(cli).items():
                setattr(model_args, k, v)
        except Exception:
            pass

    # Override with saved config — this wins for architecture params
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(model_args, k, v)
        print(f"Loaded config: {config_path}")
    else:
        print(f"WARNING: config not found at {config_path}, using defaults")

    # ── 3. Build model with correct architecture ──────────────────────
    model = NeuralNetwork(model_args)
    model.args = model_args

    # ── 4. Load weights ───────────────────────────────────────────────
    raw = np.load(model_path, allow_pickle=True)

    # np.save on a list-of-dicts gives shape (N,) object array
    # np.save on a plain dict gives a 0-d object array → use .item()
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            weights = raw.item()          # was saved as a dict
        else:
            weights = raw.tolist()        # was saved as a list of dicts
    else:
        weights = raw

    model.set_weights(weights)

    # ── 5. Sanity-check: run one sample through ───────────────────────
    dummy = np.zeros((1, 784), dtype=np.float64)
    try:
        out = model.forward(dummy)
        assert out.shape == (1, 10), f"Bad output shape: {out.shape}"
    except Exception as e:
        print(f"WARNING: forward-pass sanity check failed: {e}")

    return model


def run_inference():
    args = parse_arguments()

    model = load_model(args.model_path, args.config_path)

    # Use dataset from the loaded config, but allow CLI override
    dataset_name = getattr(model.args, "dataset", args.dataset)
    _, _, _, _, x_test, y_test = load_data(dataset_name)

    logits = model.forward(x_test)
    preds  = np.argmax(logits, axis=1)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds,    average="macro", zero_division=0)
    f1   = f1_score(y_test, preds,        average="macro", zero_division=0)

    print("\nTest Set Evaluation")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return f1


if __name__ == "__main__":
    run_inference()