import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import argparse
import json
import os
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from data_utils         import load_data, get_batches

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _winit(project, group, config, name):
    if not _WANDB: return
    try:
        wandb.init(project=project, group=group, config=config, name=name,
                   reinit=True, settings=wandb.Settings(init_timeout=30))
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",        type=int,   default=20)
    p.add_argument("-b",   "--batch_size",    type=int,   default=32)
    p.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                   choices=["mean_squared_error", "cross_entropy"])
    p.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0.0001)
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",
                   default=[128, 128, 128])
    p.add_argument("-a",   "--activation",    type=str,   nargs="+",
                   default=["relu", "relu", "relu"],
                   choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                   choices=["random", "xavier"])
    p.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment_1")
    p.add_argument("-wg",  "--wandb_group",   type=str,   default="general")
    p.add_argument("--model_path",  type=str, default="best_model.npy")
    p.add_argument("--config_path", type=str, default="best_config.json")
    return p.parse_args()


def train():
    args = parse_args()

    model_path  = os.path.join(SRC_DIR, args.model_path)
    config_path = os.path.join(SRC_DIR, args.config_path)

    n = max(1, args.num_layers)
    args.hidden_size = (args.hidden_size + [args.hidden_size[-1]] * n)[:n]
    args.activation  = (args.activation  + [args.activation[-1]]  * n)[:n]

    _winit(args.wandb_project, args.wandb_group, vars(args),
           f"{args.optimizer}_{args.loss}_nhl{args.num_layers}_lr{args.learning_rate}")

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)

    best_f1      = -1.0
    best_weights = None
    best_cfg     = vars(args).copy()

    print(f"\ntraining | dataset={args.dataset} | opt={args.optimizer} | "
          f"lr={args.learning_rate} | epochs={args.epochs}")
    print(f"arch: 784 -> {args.hidden_size} -> 10 | act={args.activation}")
    print(f"saving to: {model_path}")
    print("-" * 65)

    for ep in range(1, args.epochs + 1):
        total_loss  = 0.0
        num_batches = 0

        for xb, yb in get_batches(x_train, y_train, args.batch_size):
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)
            model.backward(logits, yb)
            model.update(args.learning_rate)
            total_loss  += loss
            num_batches += 1

        train_loss = total_loss / num_batches

        val_logits = model.forward(x_val)
        val_loss   = model.compute_loss(val_logits, y_val)
        val_preds  = np.argmax(val_logits, axis=1)
        val_acc    = float(np.mean(val_preds == y_val))

        tr_preds = np.argmax(model.forward(x_train[:5000]), axis=1)
        tr_acc   = float(np.mean(tr_preds == y_train[:5000]))

        test_preds = np.argmax(model.forward(x_test), axis=1)
        test_acc   = float(np.mean(test_preds == y_test))
        test_f1    = f1_score(y_test, test_preds, average="macro", zero_division=0)

        grad_norm = float(np.linalg.norm(model.layers[0].grad_W))

        print(f"ep {ep:02d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
              f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f}")

        _wlog({
            "epoch":        ep,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "train_acc":    tr_acc,
            "val_acc":      val_acc,
            "test_acc":     test_acc,
            "test_f1":      test_f1,
            "grad_norm_l0": grad_norm,
        })

        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()
            best_cfg["best_test_f1"] = float(best_f1)
            best_cfg["best_epoch"]   = ep
            print(f"  -> new best f1: {best_f1:.4f}")

    np.save(model_path, best_weights)

    cfg_out = {k: list(v) if isinstance(v, (list, np.ndarray)) else v
               for k, v in best_cfg.items()}
    with open(config_path, "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"\nbest test f1  : {best_f1:.4f}")
    print(f"saved model   : {model_path}")
    print(f"saved config  : {config_path}")
    _wfinish()


if __name__ == "__main__":
    train()