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

    # CHANGE 1: Make sure to unpack x_test and y_test!
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
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

        # CHANGE 2: Calculate Training Accuracy
        train_logits = model.forward(x_train)
        train_preds = np.argmax(train_logits, axis=1)
        train_acc = np.mean(train_preds == y_train)

        # Validation Accuracy
        val_logits = model.forward(x_val)
        val_oh = one_hot(y_val)
        val_loss = model.compute_loss(val_logits, val_oh)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == y_val)
        
        # CHANGE 3: Calculate Test Accuracy
        test_logits = model.forward(x_test)
        test_preds = np.argmax(test_logits, axis=1)
        test_acc = np.mean(test_preds == y_test)

        # Log EVERYTHING to W&B
        wandb.log({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_acc": test_acc
        })