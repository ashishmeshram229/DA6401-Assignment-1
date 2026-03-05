import numpy as np
import argparse
import json
import os
import wandb
from sklearn.metrics import f1_score, accuracy_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data, get_batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",       choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0005)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   nargs="+", default=["relu", "relu", "relu"],
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-wi",  "--weight_init",   type=str,   default="xavier", choices=["random", "xavier", "zeros"])
    parser.add_argument("-wp",  "--wandb_project", type=str,   default="da6401_assignment_1")
    parser.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    parser.add_argument("--model_path",  type=str, default="src/best_model.npy")
    parser.add_argument("--config_path", type=str, default="src/best_config.json")
    return parser.parse_args()


def compute_accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=1) == labels)


def train():
    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        config=vars(args),
        name=f"{args.optimizer}_{args.loss}_L{args.num_layers}_lr{args.learning_rate}"
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    # ------------------------------------------------------------------ #
    # NOTE: Pass INTEGER labels (y_train) directly — NOT one-hot.         #
    # NeuralNetwork.compute_loss and backward both handle int labels now. #
    # ------------------------------------------------------------------ #
    model = NeuralNetwork(args)

    best_test_f1   = -1.0
    best_weights   = None
    best_config    = vars(args).copy()

    print(f"\nTraining | dataset={args.dataset} | opt={args.optimizer} | "
          f"lr={args.learning_rate} | epochs={args.epochs} | layers={args.num_layers} | "
          f"hidden={args.hidden_size} | act={args.activation}")

    for ep in range(1, args.epochs + 1):

        total_loss = 0.0
        n_batches  = 0

        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=42 + ep):
            # Forward
            logits = model.forward(xb)

            # Loss  — pass INTEGER labels, not one-hot
            loss = model.compute_loss(logits, yb)

            # Backward — pass INTEGER labels, not one-hot
            model.backward(logits, yb)

            # Log first-layer gradient norm for W&B report Q2.4
            grad_norm = np.linalg.norm(model.layers[0].grad_W)
            wandb.log({"first_layer_grad_norm": grad_norm})

            # Parameter update
            model.update(args.learning_rate)

            total_loss += loss
            n_batches  += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # ---- Validation metrics ----
        val_logits  = model.forward(x_val)
        val_loss    = model.compute_loss(val_logits, y_val)
        val_acc     = compute_accuracy(val_logits, y_val)
        val_preds   = np.argmax(val_logits, axis=1)
        val_f1      = f1_score(y_val, val_preds, average="macro", zero_division=0)

        # ---- Train subset metrics ----
        tr_logits   = model.forward(x_train[:5000])
        tr_acc      = compute_accuracy(tr_logits, y_train[:5000])

        # ---- Test metrics — used for model selection ----
        test_logits = model.forward(x_test)
        test_preds  = np.argmax(test_logits, axis=1)
        test_f1     = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_acc    = accuracy_score(y_test, test_preds)

        print(f"Epoch {ep:02d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | test_f1={test_f1:.4f}")

        # Dead-neuron fraction (only meaningful for ReLU)
        dead_frac = float(np.mean(model.layers[0].a == 0.0))

        wandb.log({
            "epoch":               ep,
            "train_loss":          avg_train_loss,
            "val_loss":            val_loss,
            "val_acc":             val_acc,
            "val_f1":              val_f1,
            "train_acc":           tr_acc,
            "test_acc":            test_acc,
            "test_f1":             test_f1,
            "dead_neuron_fraction": dead_frac,
        })

        # Save best model by test F1 (as required by assignment)
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_weights = model.get_weights()
            best_config["best_test_f1"] = float(best_test_f1)
            best_config["best_epoch"]   = ep
            print(f"  ✓ New best test F1: {best_test_f1:.4f}  (epoch {ep})")

    # ---- Persist best model ----
    os.makedirs("src", exist_ok=True)
    model_path  = args.model_path
    config_path = args.config_path

    np.save(model_path, best_weights, allow_pickle=True)
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest Test F1 : {best_test_f1:.4f}")
    print(f"Saved model  → {model_path}")
    print(f"Saved config → {config_path}")

    wandb.finish()


if __name__ == "__main__":
    train()