import argparse
import json
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_arguments(args_list=None):
    p = argparse.ArgumentParser()
    p.add_argument("-d",    "--dataset",      type=str,   default="mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",    "--epochs",       type=int,   default=20)
    p.add_argument("-b",    "--batch_size",   type=int,   default=32)
    p.add_argument("-l",    "--loss",         type=str,   default="cross_entropy",
                   choices=["mean_squared_error", "cross_entropy"])
    p.add_argument("-o",    "--optimizer",    type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",   "--learning_rate",type=float, default=0.001)
    p.add_argument("-wd",   "--weight_decay", type=float, default=0.0001)
    p.add_argument("-nhl",  "--num_layers",   type=int,   default=3)
    p.add_argument("-sz",   "--hidden_size",  type=int,   nargs="+",
                   default=[128, 128, 128])
    p.add_argument("-a",    "--activation",   type=str,   nargs="+",
                   default=["relu", "relu", "relu"],
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i",  "--weight_init",  type=str,   default="xavier",
                   choices=["random", "xavier"])
    p.add_argument("-w_p",  "--wandb_project",type=str,   default="da6401_assignment_1")
    p.add_argument("--model_path",  type=str, default="best_model.npy")
    p.add_argument("--config_path", type=str, default="best_config.json")
    if args_list is not None:
        return p.parse_args(args_list)
    return p.parse_args()

parse_args = parse_arguments


def load_model(model_path, config_path=None):
    # Find config — try given path, then src/, then alongside model
    candidates = [
        config_path,
        os.path.join(SRC_DIR, "best_config.json"),
        os.path.join(os.path.dirname(os.path.abspath(model_path)), "best_config.json"),
        "best_config.json",
    ]
    found_cfg = next((c for c in candidates if c and os.path.exists(c)), None)

    # Build args from defaults
    model_args = parse_arguments([])

    # Override with config file (controls architecture)
    if found_cfg:
        print(f"Loaded config: {found_cfg}")
        with open(found_cfg) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(model_args, k):
                setattr(model_args, k, v)
    else:
        print("WARNING: best_config.json not found — using CLI defaults")

    # Build model
    model = NeuralNetwork(model_args)
    model.args = model_args

    # Load weights — assignment says: np.load(..., allow_pickle=True).item()
    raw = np.load(os.path.join(SRC_DIR, model_path)
                  if not os.path.isabs(model_path) and not os.path.exists(model_path)
                  else model_path,
                  allow_pickle=True)
    weights = raw.item() if raw.ndim == 0 else raw.tolist()
    model.set_weights(weights)

    # Sanity check
    dummy = np.zeros((1, 784), dtype=np.float64)
    out = model.forward(dummy)
    assert out.shape == (1, 10), \
        f"Bad output shape: {out.shape} — weight/config mismatch!"

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
    rec  = recall_score(y_test,   preds, average="macro", zero_division=0)
    f1   = f1_score(y_test,       preds, average="macro", zero_division=0)

    print(f"\nTest Set Evaluation")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return f1


if __name__ == "__main__":
    run_inference()