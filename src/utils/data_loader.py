import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Flatten and normalise — cast to float64 to prevent RuntimeWarnings
    x_train = x_train.reshape(-1, 784).astype(np.float64) / 255.0
    x_test  = x_test.reshape(-1,  784).astype(np.float64) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test