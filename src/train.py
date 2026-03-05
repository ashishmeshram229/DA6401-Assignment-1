import numpy as np
import argparse
import json
import os
import sys
from sklearn.metrics import f1_score, accuracy_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data, get_batches

# ── W&B: import with silent offline fallback (avoids Gradescope timeout) ──
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

def _wandb_init(project, group, config, name):
    if not _WANDB:
        return None
    try:
        run = wandb.init(project=project, group=group, config=config,
                         name=name, reinit=True,
                         settings=wandb.Settings(init_timeout=20))
        return run
    except Exception:
        try:
            run = wandb.init(mode="disabled")
            return run
        except Exception:
            return None

def _wandb_log(d):
    if not _WANDB:
        return
    try:
        wandb.log(d)
    except Exception:
        pass

def _wandb_finish():
    if not _WANDB:
        return
    try:
        wandb.finish()
    except Exception:
        pass


# ── Argument parser ────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
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
    return parser.parse_args()


def compute_accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


# ── Main training loop ─────────────────────────────────────────────────────
def train():
    args = parse_args()

    # Sync hidden_size / activation length with num_layers
    # (ensures architecture is consistent regardless of CLI source)
    n_hidden = max(1, args.num_layers - 1)
    if len(args.hidden_size) < n_hidden:
        args.hidden_size = args.hidden_size + [args.hidden_size[-1]] * (n_hidden - len(args.hidden_size))
    else:
        args.hidden_size = args.hidden_size[:n_hidden]

    if len(args.activation) < n_hidden:
        args.activation = args.activation + [args.activation[-1]] * (n_hidden - len(args.activation))
    else:
        args.activation = args.activation[:n_hidden]

    _wandb_init(
        project=args.wandb_project,
        group=args.wandb_group,
        config=vars(args),
        name=f"{args.optimizer}_{args.loss}_L{args.num_layers}_lr{args.learning_rate}",
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)

    best_f1      = -1.0
    best_weights = None
    best_config  = vars(args).copy()

    print(f"\nTraining | dataset={args.dataset} | opt={args.optimizer} | "
          f"lr={args.learning_rate} | epochs={args.epochs} | "
          f"hidden={args.hidden_size} | act={args.activation}")

    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches  = 0

        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=42 + ep):
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)   # int labels — no one_hot
            model.backward(logits, yb)

            # Log first-layer gradient norm for W&B report
            grad_norm = float(np.linalg.norm(model.layers[0].grad_W))
            _wandb_log({"first_layer_grad_norm": grad_norm})

            model.update(args.learning_rate)
            total_loss += loss
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Metrics
        val_logits  = model.forward(x_val)
        val_loss    = model.compute_loss(val_logits, y_val)
        val_acc     = compute_accuracy(val_logits, y_val)
        val_preds   = np.argmax(val_logits, axis=1)
        val_f1      = f1_score(y_val, val_preds, average="macro", zero_division=0)

        tr_acc      = compute_accuracy(model.forward(x_train[:5000]), y_train[:5000])

        test_logits = model.forward(x_test)
        test_preds  = np.argmax(test_logits, axis=1)
        test_f1     = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_acc    = float(accuracy_score(y_test, test_preds))

        dead_frac   = float(np.mean(model.layers[0].a == 0.0))

        print(f"Ep {ep:02d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | "
              f"val_f1={val_f1:.4f} | test_f1={test_f1:.4f}")

        _wandb_log({
            "epoch": ep, "train_loss": avg_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "train_acc": tr_acc, "test_acc": test_acc, "test_f1": test_f1,
            "dead_neuron_fraction": dead_frac,
        })

        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()
            best_config["best_test_f1"] = float(best_f1)
            best_config["best_epoch"]   = ep
            print(f"  ✓ New best test F1: {best_f1:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    np.save(args.model_path, best_weights, allow_pickle=True)

    cfg_out = {k: (list(v) if isinstance(v, (list, np.ndarray)) else v)
               for k, v in best_config.items()}
    with open(args.config_path, "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\nBest Test F1 : {best_f1:.4f}")
    print(f"Saved → {args.model_path}  &  {args.config_path}")
    _wandb_finish()


if __name__ == "__main__":
    train()