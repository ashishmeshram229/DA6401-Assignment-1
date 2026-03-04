import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ann.neural_network import NeuralNetwork
from data_utils import load_data



def run_error_analysis():
    # Load config and best weights
    with open("src/best_config.json", "r") as f:
        cfg = json.load(f)
        


    class A: pass
    args = A()
    for k, v in cfg.items(): setattr(args, k, v)
    

    _, _, _, _, x_test, y_test = load_data(args.dataset)
    
    model = NeuralNetwork(args)
    weights = np.load("src/best_model.npy", allow_pickle=True)
    model.set_weights(weights)
    


    # Predict
    logits = model.forward(x_test)
    preds = np.argmax(logits, axis=1)
    

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Best Model")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png!")
    


    # 2. Visualize Failures

    errors = np.where(preds != y_test)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Where the Model Failed (True Label vs Prediction)", fontsize=16)
    

    for i, ax in enumerate(axes.flatten()):
        if i < len(errors):
            idx = errors[i]
            # Reshape back to 28x28 for visualization
            img = x_test[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {y_test[idx]} | Pred: {preds[idx]}", color="red")
        ax.axis('off')
        
        
    plt.tight_layout()
    plt.savefig("model_failures.png")
    print("Saved model_failures.png!")

if __name__ == "__main__":
    run_error_analysis()