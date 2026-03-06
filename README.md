# DA6401 — Assignment 1 · Multi-Layer Perceptron from Scratch

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![wandb](https://img.shields.io/badge/Weights_%26_Biases-Tracked-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/ashishmeshram229-indian-institute-of-technology-madras/da6401_assignment_1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjA5MTI2Mw?accessToken=05w3uydugj3p158y5nda30qyluq4q1g3abpx7v3rwzq0c0gnwzz5u84g1wduxyld)
![IIT Madras](https://img.shields.io/badge/IIT%20Madras-DA6401-8B0000?style=for-the-badge)

**A fully NumPy-based neural network implementation — no PyTorch, no TensorFlow.**

[📊 W&B Report](https://wandb.ai/ashishmeshram229-indian-institute-of-technology-madras/da6401_assignment_1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjA5MTI2Mw?accessToken=05w3uydugj3p158y5nda30qyluq4q1g3abpx7v3rwzq0c0gnwzz5u84g1wduxyld) · [💻 GitHub Repo](https://github.com/ashishmeshram229/DA6401-Assignment-1)

</div>

---

## Overview

This repository contains a **feed-forward neural network built entirely from scratch using NumPy** for the DA6401 Introduction to Deep Learning course at IIT Madras. No deep learning framework is used for training — every forward pass, backward pass, gradient calculation, and weight update is hand-implemented.

The implementation is trained and evaluated on **MNIST** and **Fashion-MNIST** datasets, achieving **97.86% test accuracy** on MNIST with hyperparameters tuned via Weights & Biases sweeps.

---

## Results

| Dataset | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| MNIST (test) | **97.86%** | **97.85%** | **97.86%** | **97.84%** |
| Fashion-MNIST (test) | ~88% | ~88% | ~88% | ~88% |

**Best configuration (MNIST):**

| Hyperparameter | Value |
|---|---|
| Optimizer | RMSProp |
| Learning Rate | 0.001 |
| Hidden Layers | 3 |
| Hidden Size | 128 × 128 × 128 |
| Activation | ReLU |
| Weight Init | Xavier |
| Loss | Cross-Entropy |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Epochs | 20 |

---

## Project Structure

```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py              # Package init — adds src/ to sys.path
│   │   ├── activations.py           # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── layer.py                 # Single fully-connected layer (forward + backward)
│   │   ├── neural_network.py        # Full MLP model (train, evaluate, get/set weights)
│   │   ├── losses.py                # Cross-entropy and MSE losses + gradients
│   │   └── optimizers.py            # SGD, Momentum, NAG, RMSProp
│   ├── utils/
│   │   └── data_loader.py           # MNIST / Fashion-MNIST loading + preprocessing
│   ├── train.py                     # Training script with CLI + W&B logging
│   ├── inference.py                 # Inference script — load model and evaluate
│   ├── grad_check.py                # Numerical gradient verification
│   ├── best_model.npy               # Saved best model weights
│   └── best_config.json             # Saved best hyperparameter config
├── README.md
└── requirements.txt
```

---

## Implementation Details

### Architecture

A configurable feed-forward MLP: `784 → [hidden layers] → 10`

- **Input**: Flattened 28×28 image → 784-dimensional vector, normalised to [0, 1]
- **Hidden layers**: Fully connected with configurable activation (ReLU / Sigmoid / Tanh)
- **Output**: 10 logits (one per class), converted to probabilities via Softmax

### Activations

| Activation | Forward | Gradient |
|---|---|---|
| ReLU | `max(0, z)` | `1 if z > 0 else 0` |
| Sigmoid | `1 / (1 + e^{-z})` | `σ(z) · (1 − σ(z))` |
| Tanh | `tanh(z)` | `1 − tanh²(z)` |

### Loss Functions

| Loss | Formula |
|---|---|
| Cross-Entropy | `−(1/N) Σ log(p[y])` |
| MSE | `(1/N) Σ (softmax(logits) − one_hot(y))²` |

> Gradients are divided by batch size **once** inside the loss gradient — not inside layer backward passes. This is critical for correct numerical gradient checks.

### Optimizers

| Optimizer | Key Formula |
|---|---|
| SGD | `W ← W − lr · ∇W` |
| Momentum | `v ← β·v + lr·∇W` &nbsp;&nbsp; `W ← W − v` |
| NAG | Look-ahead gradient at `W − β·v` |
| RMSProp | `s ← β·s + (1−β)·∇W²` &nbsp;&nbsp; `W ← W − lr·∇W / √(s + ε)` |

### Weight Initialisation

- **Xavier (Glorot)**: `U[−√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out))]`
- **Random**: `N(0, 0.01²)`

---

## Setup

### Prerequisites

```bash
pip install numpy scikit-learn keras wandb
```

### Clone

```bash
git clone https://github.com/ashishmeshram229/DA6401-Assignment-1
cd DA6401-Assignment-1/src
```

---

## Usage

### Train

```bash
cd src

python3 train.py -d mnist -e 20 -b 32 -o rmsprop -lr 0.001 \
  -nhl 3 -sz 128 128 128 -a relu \
  -w_i xavier -l cross_entropy -wd 0.0001 \
  -w_p DL-AS1-WANDB --model_save_path best_model.npy
```

Train without W&B:

```bash
python3 train.py -d fashion_mnist -e 20 -b 32 -o rmsprop -lr 0.001 \
  -nhl 3 -sz 128 128 128 -a relu \
  -w_i xavier -l cross_entropy -wd 0.0001 \
  --no_wandb --model_save_path best_model.npy
```

### Inference

```bash
# Automatically reads best_config.json — no extra args needed
python3 inference.py

# Or specify explicitly
python3 inference.py -d mnist --model_path best_model.npy
```

### Gradient Check

```bash
python3 grad_check.py
# Expected output: Max gradient difference: ~2.74e-10  →  PASS
```

---

## CLI Reference

### `train.py`

| Argument | Short | Default | Description |
|---|---|---|---|
| `--dataset` | `-d` | `mnist` | Dataset: `mnist` or `fashion_mnist` |
| `--epochs` | `-e` | `20` | Number of training epochs |
| `--batch_size` | `-b` | `32` | Mini-batch size |
| `--optimizer` | `-o` | `rmsprop` | `sgd` / `momentum` / `nag` / `rmsprop` |
| `--learning_rate` | `-lr` | `0.001` | Learning rate |
| `--weight_decay` | `-wd` | `0.0001` | L2 regularisation coefficient |
| `--num_layers` | `-nhl` | `3` | Number of hidden layers |
| `--hidden_size` | `-sz` | `128 128 128` | Neurons per hidden layer |
| `--activation` | `-a` | `relu` | `relu` / `sigmoid` / `tanh` |
| `--weight_init` | `-w_i` | `xavier` | `xavier` or `random` |
| `--loss` | `-l` | `cross_entropy` | `cross_entropy` or `mse` |
| `--wandb_project` | `-w_p` | — | W&B project name |
| `--no_wandb` | — | `False` | Disable W&B logging |
| `--model_save_path` | — | `best_model.npy` | Path to save best model |

### `inference.py`

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `best_model.npy` | Path to saved `.npy` weights |
| `-d` | `mnist` | Dataset to evaluate on |

---

## Experiment Tracking

All training runs are tracked on Weights & Biases.

📊 **[View Full W&B Report →](https://wandb.ai/ashishmeshram229-indian-institute-of-technology-madras/da6401_assignment_1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjA5MTI2Mw?accessToken=05w3uydugj3p158y5nda30qyluq4q1g3abpx7v3rwzq0c0gnwzz5u84g1wduxyld)**

Logged metrics include:
- `train_loss`, `train_acc` per epoch
- `val_loss`, `val_acc`, `val_f1` per epoch
- `test_acc`, `test_f1` at end of training

---

## Key Design Decisions

**Gradient convention** — Division by batch size `N` happens exactly once, inside the loss gradient function (`cross_entropy_grad`, `mse_grad`). Layer `backward()` does NOT divide again. This ensures numerical gradient checks pass with error < 1e-7.

**Float64 throughout** — All weights, inputs, and activations are cast to `float64` at computation time. This eliminates NumPy overflow/invalid-value warnings that occur when float32 MNIST data mixes with float64 weights in matmul.

**`sys.path` self-healing** — Every file in `ann/` inserts `src/` into `sys.path` using its own `__file__` path. This makes the package importable regardless of whether you run `python3 train.py` from inside `src/` or `python3 src/train.py` from the project root.

**set_weights robustness** — `set_weights()` handles `dict`, `list`, `tuple`, and numpy 0-d object arrays (produced by `np.save` / `np.load`), so the autograder's `dummy_model.npy` in any format loads correctly.

---

## Author

**Ashish Meshram (DA25M016)**
M.Tech · IIT Madras
[GitHub](https://github.com/ashishmeshram229/DA6401-Assignment-1) · [W&B](https://wandb.ai/ashishmeshram229-indian-institute-of-technology-madras/da6401_assignment_1/reports/DA6401-Assignment-1-Multi-Layer-Perceptron--VmlldzoxNjA5MTI2Mw?accessToken=05w3uydugj3p158y5nda30qyluq4q1g3abpx7v3rwzq0c0gnwzz5u84g1wduxyld)

---

<div align="center">
<sub>DA6401 · Introduction to Deep Learning · IIT Madras · 2025–26</sub>
</div>
