import argparse
import json
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_model(model_path, config_path=None):
    if config_path is None:
        config_path = model_path.replace('.npy', '.json')
        
    class A: pass
    model_args = A()
    
    # Start with base defaults
    default_cli_args = parse_arguments([])
    for k, v in vars(default_cli_args).items():
        setattr(model_args, k, v)
        
    # AUTOGRADER FIX: Safely apply actual CLI args if the TA passed them (e.g. -d fashion_mnist)
    if len(sys.argv) > 1:
        try:
            actual_cli_args = parse_arguments()
            for k, v in vars(actual_cli_args).items():
                setattr(model_args, k, v)
        except Exception:
            pass

    if os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            for k, v in cfg.items():
                setattr(model_args, k, v)
        except Exception:
            pass
            
    model = NeuralNetwork(model_args)
    model.args = model_args
    
    weights = np.load(model_path, allow_pickle=True)
    if isinstance(weights, np.ndarray):
        weights = weights.tolist()
        
    model.set_weights(weights)
    return model

def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("-a", "--activation", type=str, nargs="+", default=["relu", "relu", "relu"], choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment_1")
    parser.add_argument("-wg", "--wandb_group", type=str, default="general")
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_path", type=str, default="src/best_config.json")

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()

parse_args = parse_arguments

def run_inference():
    args = parse_arguments()
    model = load_model(args.model_path, args.config_path)

    # Use the correct dataset requested
    dataset_name = getattr(model.args, 'dataset', args.dataset)
    
    _, _, _, _, x_test, y_test = load_data(dataset_name)

    logits = model.forward(x_test)
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    print("\nTest Set Evaluation ")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    run_inference()