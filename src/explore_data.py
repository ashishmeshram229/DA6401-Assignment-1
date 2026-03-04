import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

def explore():
    (x_train, y_train), _ = mnist.load_data()
    classes = np.unique(y_train)
    
    fig, axes = plt.subplots(len(classes), 5, figsize=(10, 15))
    fig.suptitle("MNIST Dataset: 5 Samples per Class", fontsize=16)
    
    for c in classes:
        # Find the first 5 images that match class 'c'
        idx = np.where(y_train == c)[0][:5]
        for i, image_idx in enumerate(idx):
            ax = axes[c, i]
            ax.imshow(x_train[image_idx], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Class {c}", loc='left')
                
    plt.tight_layout()
    plt.savefig("data_exploration.png")
    print("Saved data_exploration.png!")

if __name__ == "__main__":
    explore()