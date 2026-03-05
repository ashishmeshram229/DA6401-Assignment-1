import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=32)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse", "mean_squared_error"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0001)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   nargs="+", default=["relu"],
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment1")
    parser.add_argument("--wandb_entity",          type=str,   default=None)
    parser.add_argument("--model_path",            type=str,   default="best_model.npy")
    parser.add_argument("--config_path",           type=str,   default="best_config.json")

    return parser.parse_args()


def load_config_into_args(args):
    """
    Override args with values from best_config.json so the model architecture
    always matches what was saved — regardless of CLI defaults.
    """
    config_path = args.config_path

    # also look next to the model file if not found
    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)), "best_config.json"
        )

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
        print(f"Loaded config from: {config_path}")
    else:
        print("WARNING: best_config.json not found — using CLI defaults. "
              "Architecture may not match saved weights.")

    return args


def main():
    args = parse_arguments()

    # ── Load architecture from saved config ──────────────────────────────────
    args = load_config_into_args(args)

    # Normalise activation to single string
    if isinstance(args.activation, list):
        args.activation = args.activation[0]

    # Normalise loss name
    if args.loss == "mean_squared_error":
        args.loss = "mse"

    # Pad / trim hidden_size to match num_layers
    if not isinstance(args.hidden_size, list):
        args.hidden_size = [args.hidden_size] * args.num_layers
    if len(args.hidden_size) < args.num_layers:
        args.hidden_size = args.hidden_size + [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    print(f"Dataset  : {args.dataset}")
    print(f"Layers   : {args.num_layers}  Hidden: {args.hidden_size}  Act: {args.activation}")

    _, _, _, _, X_test, y_test = load_data(args.dataset)

    model   = NeuralNetwork(args)
    weights = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weights)

    results = model.evaluate(X_test, y_test)

    print(f"\nAccuracy  : {results['accuracy']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Loss      : {results['loss']:.4f}")

    return results


if __name__ == "__main__":
    main()