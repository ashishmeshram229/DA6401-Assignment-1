import argparse
import json
import sys
import os
import numpy as np
import uuid



from ann.neural_network import NeuralNetwork
from utils.data_loader  import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a neural network on MNIST / Fashion-MNIST")

    parser.add_argument("-d",   "--dataset",        type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",         type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=32)
    parser.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse", "mean_squared_error"])
    parser.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=1e-3)
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0001)
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+",
                        default=[128, 128, 128])
    # accepts single OR multiple values: -a relu  OR  -a relu relu relu
    parser.add_argument("-a",   "--activation",     type=str,   nargs="+",
                        default=["relu"],
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",    type=str,   default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project",  type=str,   default="DL-AS1-WANDB")
    parser.add_argument("--wandb_entity",           type=str,   default=None)
    parser.add_argument("--no_wandb",               action="store_true", default=False)
    parser.add_argument("--model_save_path",        type=str,   default="best_model.npy")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Normalise activation → always a single string
    if isinstance(args.activation, list):
        args.activation = args.activation[0]

    # Normalise loss name
    if args.loss == "mean_squared_error":
        args.loss = "mse"

    # Pad / trim hidden_size list to match num_layers
    if len(args.hidden_size) < args.num_layers:
        args.hidden_size = (args.hidden_size +
                            [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size)))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    print(f"Dataset : {args.dataset}")
    print(f"Arch    : num_layers={args.num_layers}  hidden={args.hidden_size}"
          f"  act={args.activation}  opt={args.optimizer}  lr={args.learning_rate}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project  = args.wandb_project,
                entity   = args.wandb_entity,
                config   = vars(args),
                name     = f"{args.optimizer}_{args.loss}_nhl{args.num_layers}",
                group    = f"train_{args.dataset}",
                reinit   = "finish_previous",
                id       = str(uuid.uuid4()),
            )
        except Exception as e:
            print(f"wandb init failed: {e}. Continuing without wandb.")

    model = NeuralNetwork(args)
    best_weights = model.train(
        X_train, y_train,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        X_val      = X_val,
        y_val      = y_val,
        wandb_run  = run,
    )

    if best_weights is not None:
        model.set_weights(best_weights)

    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nTest | acc: {test_metrics['accuracy']:.4f}"
          f" | f1: {test_metrics['f1']:.4f}"
          f" | precision: {test_metrics['precision']:.4f}"
          f" | recall: {test_metrics['recall']:.4f}")

    if run is not None:
        run.log({"test_acc": test_metrics["accuracy"],
                 "test_f1":  test_metrics["f1"]})

    # ── Save model weights ────────────────────────────────────────────────────
    save_path = os.path.abspath(args.model_save_path)
    weights   = model.get_weights()
    np.save(save_path, weights)
    print(f"Model saved  : {save_path}")

    # ── Save config (same folder as model) ───────────────────────────────────
    config_path = os.path.join(os.path.dirname(save_path), "best_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved : {config_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()