
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name):
    # Load MNIST or Fashion-MNIST dataset, normalize pixel values to [0, 1], and split training data into train and validation sets (90% train, 10% val)
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()


    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    x_train = x_train.reshape(-1, 784).astype(np.float64) / 255.0

    x_test  = x_test.reshape(-1, 784).astype(np.float64)  / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )


    return x_train, y_train, x_val, y_val, x_test, y_test



def one_hot(y, num_classes=10): # Convert class labels (as integers) to one-hot encoded vectors (used for MSE loss)

    m  = y.shape[0]
    oh = np.zeros((m, num_classes), dtype=np.float64)
    oh[np.arange(m), y.astype(int)] = 1.0

    return oh


def get_batches(x, y, batch_size):
    # Generator function to yield batches of data (used for training loop in neural_network.py)
  
    m   = x.shape[0]
    idx = np.random.permutation(m)
    x   = x[idx]
    y   = y[idx]
    
    for start in range(0, m, batch_size):
        end = start + batch_size
        yield x[start:end], y[start:end]
