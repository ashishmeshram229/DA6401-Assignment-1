# DA6401 Assignment 1  
## Multilayer Perceptron (MLP) From Scratch Using NumPy  
### *Ashish Meshram*  

This project implements a full **Multilayer Perceptron classifier** completely **from scratch** using **NumPy only**, following all requirements of **IIT-M DA6401 – Deep Learning Assignment 1**.

The implementation supports:

- Multiple hidden layers  
- Per-layer activation functions  
- Xavier/Random weight initialization  
- Cross-Entropy / MSE loss  
- SGD, Momentum, NAG, RMSProp  
- L2 regularization (correct gradient form)  
- Forward + Backward propagation manually coded  
- MNIST & Fashion-MNIST datasets  
- W&B experiment logging + hyperparameter sweeps  
- All assignment questions Q1–Q9 implemented with plots  

A full W&B report + local PNG plots are generated for the final assignment submission.

---

# 📂 Project Structure


src/
│
├── ann/
│ ├── activations.py # ReLU, Sigmoid, Tanh + derivatives
│ ├── layer.py # Single dense layer with per-layer optimizer
│ ├── losses.py # CE + MSE implemented on logits
│ ├── optimizers.py # SGD, Momentum, NAG, RMSProp
│ ├── neural_network.py # MLP model class (forward + backward)
│
├── data_utils.py # Dataset loading, batching, one-hot
├── train.py # Main training/validation script
├── inference.py # Evaluate saved model on test set
├── sweep.py # W&B Bayesian hyperparameter sweep
├── report_experiments.py # Q1–Q9 experiments + PNG plots
│
├── best_model.npy # Saved best weight dictionary
├── best_config.json # Saved hyperparameters for reconstruction
│
├── requirements.txt
└── README.md


---

# ⚙️ Installation

Install all required libraries:

```bash
pip install -r requirements.txt
🚀 Training the Model

Navigate to the src/ folder:

cd src
Basic training run:
python train.py \
-d fashion_mnist \
-l cross_entropy \
-o rmsprop \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-wp da6401_assignment_1
Detailed example:
python train.py \
-d mnist \
-e 20 \
-b 64 \
-l cross_entropy \
-o rgbprop \
-lr 0.001 \
-wd 0.0 \
-nhl 3 \
-sz 128 128 128 \
-a relu relu relu \
-wi xavier \
-wp da6401_assignment_1

After training, the following files are saved:

best_model.npy
best_config.json

These are used later for inference.

🔍 Inference (Testing the Best Model)
python inference.py \
--model_path best_model.npy \
--config_path best_config.json

Outputs:

Accuracy

Precision

Recall

Macro F1 Score

🎛 Hyperparameter Sweep

Runs Bayesian optimization over:

Learning rate

Batch size

Optimizer

Activation

Hidden layer size

Weight decay

Weight initialization

Run:

python sweep.py
📊 Full Assignment Experiments (Q1–Q9)

All experiments required for DA6401 Assignment 1clear are implemented in:

src/report_experiments.py

Run:

python report_experiments.py

This generates PNG files:

q1_activation_comparison.png
q2_learning_rate.png
q3_loss_functions.png
q4_optimizer_comparison.png
q5_dead_relu.png
q6_overfitting_curve.png
q7_confusion_matrix.png
q8_q9_weight_init.png