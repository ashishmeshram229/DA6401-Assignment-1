import numpy as np

import wandb
from ann.neural_network import NeuralNetwork

from data_utils import load_data, one_hot, get_batches

# Sweep Configuration


sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    

    "parameters": {
        "epochs":        {"values": [8, 10, 20]},
        "batch_size":    {"values": [32, 64, 128]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },


        "num_layers":   {"values": [2, 3, 4]},
        "hidden_size":  {"values": [64, 128, 256]},
        "activation":   {"values": ["relu", "tanh", "sigmoid"]},
        "weight_init":  {"values": ["random", "xavier"]},
        "weight_decay": {"values": [0.0, 1e-5, 1e-4]},
        "loss":         {"values": ["cross_entropy", "mean_squared_error"]},
        "dataset":      {"value": "mnist"}
    }
}



def sweep_train():


    run = wandb.init()
    c = wandb.config


    class A: pass
    args = A()

    args.dataset       = c.dataset
    args.epochs        = c.epochs
    args.batch_size    = c.batch_size
    args.loss          = c.loss
    args.optimizer     = c.optimizer
    args.learning_rate = c.learning_rate
    args.weight_decay  = c.weight_decay
    args.weight_init   = c.weight_init
    args.num_layers    = c.num_layers
    args.hidden_size   = [c.hidden_size] * c.num_layers
    args.activation    = [c.activation] * args.num_layers
    args.wandb_project = "da6401_assignment_1"

    x_train, y_train, x_val, y_val, _, _ = load_data(args.dataset)
    model = NeuralNetwork(args)





    for ep in range(1, args.epochs + 1):
        total_loss = 0
        batches = 0
        seed = 100 + ep


        for xb, yb in get_batches(x_train, y_train, args.batch_size, seed=seed):
            yb_oh = one_hot(yb)
            logits = model.forward(xb)
            loss = model.compute_loss(logits, yb_oh)
            model.backward(logits, yb_oh)
            model.update(args.learning_rate)

            total_loss += loss
            batches += 1

        train_loss = total_loss / batches


        # Validation

        val_logits = model.forward(x_val)
        val_oh = one_hot(y_val)
        val_loss = model.compute_loss(val_logits, val_oh)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == y_val)

        wandb.log({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })



def run_sweep():


    sweep_id = wandb.sweep(
        sweep_config,
        project="da6401_assignment_1"
    )


    wandb.agent(
        sweep_id,
        function=sweep_train,
        count=100
    )



if __name__ == "__main__":
    run_sweep()