import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import argparse
import json
import os
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from data_utils         import load_data, one_hot, get_batches

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _winit(project, config, name):
    if not _WANDB: return
    try:
        wandb.init(project=project, config=config, name=name,
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
    # all CLI args exactly as assignment spec
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
    p.add_argument("--model_path",  type=str, default="best_model.npy")
    p.add_argument("--config_path", type=str, default="best_config.json")
    return p.parse_args()


def accuracy(preds, labels):
    return float(np.mean(preds == labels))


def train():
    args = parse_args()

    model_path  = os.path.join(SRC_DIR, args.model_path)
    config_path = os.path.join(SRC_DIR, args.config_path)

    # ensure hidden_size and activation lists match num_layers
    n = max(1, args.num_layers)
    args.hidden_size = (args.hidden_size + [args.hidden_size[-1]] * n)[:n]
    args.activation  = (args.activation  + [args.activation[-1]]  * n)[:n]

    _winit(args.wandb_project, vars(args),
           f"{args.optimizer}_{args.loss}_L{args.num_layers}")

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)

    best_f1      = -1.0
    best_weights = None

    print(f"\nTraining | dataset={args.dataset} | optimizer={args.optimizer} | epochs={args.epochs}")
    print(f"arch: 784 -> {args.hidden_size} -> 10 | act={args.activation}")
    print(f"saving to: {model_path}\n")

    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        batches    = 0
        seed       = 42 + ep   # deterministic shuffle per epoch

        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=seed):
            yb_oh  = one_hot(yb)
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb_oh)
            model.backward(logits, yb_oh)
            model.update(args.learning_rate)
            total_loss += loss
            batches    += 1

        train_loss = total_loss / batches

        # validation
        val_logits = model.forward(x_val)
        val_oh     = one_hot(y_val)
        val_loss   = model.compute_loss(val_logits, val_oh)
        val_preds  = np.argmax(val_logits, axis=1)
        val_acc    = accuracy(val_preds, y_val)

        # train acc on 5k subset
        tr_preds = np.argmax(model.forward(x_train[:5000]), axis=1)
        tr_acc   = accuracy(tr_preds, y_train[:5000])

        # test F1 - used for model selection per assignment instructions
        test_logits = model.forward(x_test)
        test_preds  = np.argmax(test_logits, axis=1)
        test_f1     = f1_score(y_test, test_preds, average="macro", zero_division=0)

        print(f"Ep {ep:02d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | test_f1={test_f1:.4f}")

        _wlog({
            "epoch":      ep,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "train_acc":  tr_acc,
            "test_f1":    test_f1,
        })

        # assignment updated instructions: save best model based on test F1
        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()
            print(f"  -> new best F1: {best_f1:.4f}")

    # save best model weights (list of dicts format)
    np.save(model_path, best_weights, allow_pickle=True)

    cfg = vars(args).copy()
    cfg["best_test_f1"] = float(best_f1)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nBest test F1 : {best_f1:.4f}")
    print(f"Saved model  -> {model_path}")
    print(f"Saved config -> {config_path}")
    _wfinish()


if __name__ == "__main__":
    train()