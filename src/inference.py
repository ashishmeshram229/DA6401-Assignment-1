import argparse
import json
import sys
import os
import numpy as np


from ann.neural_network import NeuralNetwork
from utils.data_loader  import load_data



def parse_arguments(): # Called by autograder as inference.parse_arguments(). Parses CLI arguments for inference script and returns args object.

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
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",
                        
                        default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier"])
    
    parser.add_argument("-w_p", "--wandb_project", type=str,   default="DL-AS1-WANDB")
    parser.add_argument("--wandb_entity",          type=str,   default=None)
    parser.add_argument("--model_path",            type=str,   default="best_model.npy")


    return parser.parse_args()


def load_model(model_path): # Load model weights from .npy file and return as dictionary (used by inference.py to load saved model weights for evaluation)

    data = np.load(model_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    return data


def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set and return a dictionary of metrics (logits, loss, accuracy, f1, precision, recall)
    metrics = model.evaluate(X_test, y_test)

    return {
        "logits":    metrics["logits"],
        "loss":      metrics["loss"],
        "accuracy":  metrics["accuracy"],
        "f1":        metrics["f1"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
    }


def main():

    args = parse_arguments()


    if isinstance(args.activation, list): # Normalise activation to always be a single string
        args.activation = args.activation[0]

    
    if args.loss == "mean_squared_error": # Normalise loss name to "mse" if "mean_squared_error" is given (for consistency with training script and config files)
        args.loss = "mse"

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(args.model_path)), "best_config.json")
    
    if os.path.exists(config_path):

        with open(config_path) as f:
            cfg = json.load(f)


        for key in ("num_layers", "hidden_size", "activation", "weight_init",
                    "loss", "optimizer", "learning_rate", "weight_decay", "dataset"):
            
            if key in cfg:
                setattr(args, key, cfg[key])


        print(f"Config loaded from: {config_path}")

    # Re-normalise after config load
    if isinstance(args.activation, list):

        args.activation = args.activation[0]


    if args.loss == "mean_squared_error":

        args.loss = "mse"

    # Pad / trim hidden_size
    if not isinstance(args.hidden_size, list):

        args.hidden_size = [args.hidden_size] * args.num_layers

    if len(args.hidden_size) < args.num_layers:

        args.hidden_size = (args.hidden_size +
                            [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size)))
        
    elif len(args.hidden_size) > args.num_layers: # Trim hidden_size list to match num_layers if it's too long (shouldn't happen if config is correct, but added for safety)

        args.hidden_size = args.hidden_size[:args.num_layers]

    print(f"Dataset : {args.dataset}")
    print(f"Arch    : num_layers={args.num_layers}  hidden={args.hidden_size}"
          f"  act={args.activation}")

    _, _, _, _, X_test, y_test = load_data(args.dataset)

    model   = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy  : {results['accuracy']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Loss      : {results['loss']:.4f}")

    return results



if __name__ == "__main__":
    main()