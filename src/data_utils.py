import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(name):
    name = name.lower()
    if name == "mnist":
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    elif name == "fashion_mnist":
        (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # use float64 — no overflow possible
    x_tr = x_tr.reshape(-1, 784).astype(np.float64) / 255.0
    x_te = x_te.reshape(-1, 784).astype(np.float64) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_tr, y_tr, test_size=0.1, random_state=42)

    return x_train, y_train, x_val, y_val, x_te, y_te


def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def get_batches(x, y, batch_size, seed=None):
    """Accepts optional seed — matches skeleton signature."""
    m = x.shape[0]
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(m)
    else:
        idx = np.random.permutation(m)
    x, y = x[idx], y[idx]
    for i in range(0, m, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]