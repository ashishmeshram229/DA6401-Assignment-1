import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from data_utils         import load_data

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """
    Assignment says: CLI for train.py and inference.py must be the same.
    Add best model config as default values.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",        type=int,   default=10)
    p.add_argument("-b",   "--batch_size",    type=int,   default=64)
    p.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                   choices=["mean_squared_error", "cross_entropy"])
    p.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    p.add_argument("-a",   "--activation",    type=str,   nargs="+",
                   default=["relu", "relu", "relu"],
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                   choices=["random", "xavier"])
    p.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment_1")
    p.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    p.add_argument("--model_path",  type=str,
                   default=os.path.join(SRC_DIR, "best_model.npy"))
    p.add_argument("--config_path", type=str,
                   default=os.path.join(SRC_DIR, "best_config.json"))
    return p.parse_args()

# alias - some autograder versions call parse_arguments()
parse_arguments = parse_args


def load_model(model_path, config_path=None):
    # find config
    cfg_path = config_path
    if cfg_path is None or not os.path.exists(cfg_path):
        for candidate in [
            os.path.join(os.path.dirname(os.path.abspath(model_path)), "best_config.json"),
            os.path.join(SRC_DIR, "best_config.json"),
            "best_config.json",
        ]:
            if os.path.exists(candidate):
                cfg_path = candidate
                break

    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        print("WARNING: best_config.json not found - using default args")
        cfg = {}

    # build model args
    class A: pass
    model_args = A()
    model_args.dataset       = cfg.get("dataset",       "fashion_mnist")
    model_args.loss          = cfg.get("loss",          "cross_entropy")
    model_args.optimizer     = cfg.get("optimizer",     "rmsprop")
    model_args.learning_rate = cfg.get("learning_rate", 0.001)
    model_args.weight_decay  = cfg.get("weight_decay",  0.0)
    model_args.num_layers    = cfg.get("num_layers",    3)
    model_args.hidden_size   = cfg.get("hidden_size",   [128, 128, 128])
    model_args.activation    = cfg.get("activation",    ["relu", "relu", "relu"])
    model_args.weight_init   = cfg.get("weight_init",   "xavier")
    model_args.wandb_project = cfg.get("wandb_project", "da6401_assignment_1")

    # ensure list lengths match
    n = max(1, model_args.num_layers)
    if not isinstance(model_args.hidden_size, list):
        model_args.hidden_size = [model_args.hidden_size] * n
    if not isinstance(model_args.activation, list):
        model_args.activation = [model_args.activation] * n
    model_args.hidden_size = (model_args.hidden_size + [model_args.hidden_size[-1]] * n)[:n]
    model_args.activation  = (model_args.activation  + [model_args.activation[-1]]  * n)[:n]

    model = NeuralNetwork(model_args)

    # load weights - assignment says use np.load(...).item()
    raw = np.load(model_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        weights = raw.item()
    else:
        weights = raw
    model.load_weights(weights)

    return model, model_args


def run_inference():
    args = parse_args()
    model, model_args = load_model(args.model_path, args.config_path)

    _, _, _, _, x_test, y_test = load_data(model_args.dataset)

    logits = model.forward(x_test)
    preds  = np.argmax(logits, axis=1)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test,    preds, average="macro", zero_division=0)
    f1   = f1_score(y_test,        preds, average="macro", zero_division=0)

    print("\n================= Test Set Evaluation =================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=======================================================\n")
    return f1


if __name__ == "__main__":
    run_inference()