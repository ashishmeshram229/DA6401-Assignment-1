import numpy as np
import argparse
import json
import os
import sys
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from data_utils import load_data, get_batches

# ── W&B with silent fallback (Gradescope has no internet) ─────────────────
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

def _winit(project, group, config, name):
    if not _WANDB: return
    try:
        wandb.init(project=project, group=group, config=config, name=name,
                   reinit=True, settings=wandb.Settings(init_timeout=15))
    except Exception:
        try: wandb.init(mode="disabled")
        except Exception: pass

def _wlog(d):
    if not _WANDB: return
    try: wandb.log(d)
    except Exception: pass

def _wfinish():
    if not _WANDB: return
    try: wandb.finish()
    except Exception: pass


# ── CLI ───────────────────────────────────────────────────────────────────
# Per assignment spec: defaults must be the best-performing configuration.
# Architecture: num_layers counts ALL layers (hidden + output).
# So num_layers=3 with hidden_size=[128,128] means 2 hidden + 1 output.
def parse_args():
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
    # num_layers = number of hidden layers + 1 output layer
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    # hidden_size: one value per hidden layer → len must equal num_layers-1
    p.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",
                   default=[128, 128])
    p.add_argument("-a",   "--activation",    type=str,   nargs="+",
                   default=["relu", "relu"],
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi",  "--weight_init",   type=str,   default="xavier",
                   choices=["random", "xavier", "zeros"])
    p.add_argument("-wp",  "--wandb_project", type=str,   default="da6401_assignment_1")
    p.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    p.add_argument("--model_path",  type=str, default="src/best_model.npy")
    p.add_argument("--config_path", type=str, default="src/best_config.json")
    return p.parse_args()


def train():
    args = parse_args()

    # Ensure hidden_size length matches num_layers-1
    n_hidden = max(1, args.num_layers - 1)
    while len(args.hidden_size) < n_hidden:
        args.hidden_size.append(args.hidden_size[-1])
    args.hidden_size = args.hidden_size[:n_hidden]
    while len(args.activation) < n_hidden:
        args.activation.append(args.activation[-1])
    args.activation = args.activation[:n_hidden]

    _winit(args.wandb_project, args.wandb_group, vars(args),
           f"{args.optimizer}_{args.loss}_L{args.num_layers}_lr{args.learning_rate}")

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)

    best_f1      = -1.0
    best_weights = None
    best_cfg     = vars(args).copy()

    print(f"\nTraining | dataset={args.dataset} | opt={args.optimizer} | "
          f"lr={args.learning_rate} | epochs={args.epochs} | "
          f"arch=784→{args.hidden_size}→10")

    for ep in range(1, args.epochs + 1):
        total, nb = 0.0, 0

        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=42 + ep):
            logits = model.forward(xb)
            # Pass INTEGER labels directly — no one_hot conversion needed
            loss   = model.compute_loss(logits, yb)
            model.backward(logits, yb)
            _wlog({"first_layer_grad_norm": float(np.linalg.norm(model.layers[0].grad_W))})
            model.update(args.learning_rate)
            total += loss
            nb    += 1

        # Val metrics
        vl  = model.forward(x_val)
        vacc = float(np.mean(np.argmax(vl, axis=1) == y_val))
        vloss = model.compute_loss(vl, y_val)

        # Test metrics (used for model selection per assignment spec)
        tl   = model.forward(x_test)
        tp   = np.argmax(tl, axis=1)
        tf1  = f1_score(y_test, tp, average="macro", zero_division=0)
        tacc = float(np.mean(tp == y_test))

        # Train subset
        tracc = float(np.mean(np.argmax(model.forward(x_train[:5000]), axis=1) == y_train[:5000]))

        print(f"Ep {ep:02d} | loss={total/nb:.4f} | val_acc={vacc:.4f} | "
              f"test_acc={tacc:.4f} | test_f1={tf1:.4f}")

        _wlog({"epoch": ep, "train_loss": total/nb, "val_loss": vloss,
               "val_acc": vacc, "train_acc": tracc,
               "test_acc": tacc, "test_f1": tf1,
               "dead_neuron_fraction": float(np.mean(model.layers[0].a == 0.0))})

        if tf1 > best_f1:
            best_f1      = tf1
            best_weights = model.get_weights()
            best_cfg["best_test_f1"] = float(best_f1)
            best_cfg["best_epoch"]   = ep
            print(f"  ✓ New best F1: {best_f1:.4f}")

    # Save — always to src/ as required by assignment
    os.makedirs("src", exist_ok=True)
    np.save(args.model_path,  best_weights, allow_pickle=True)
    cfg_out = {k: list(v) if isinstance(v, (list, np.ndarray)) else v
               for k, v in best_cfg.items()}
    with open(args.config_path, "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\nBest Test F1 : {best_f1:.4f}")
    print(f"Saved → {args.model_path}  &  {args.config_path}")
    _wfinish()


if __name__ == "__main__":
    train()