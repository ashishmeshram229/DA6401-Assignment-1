"""
objective_functions.py
Re-exports LOSS_FN and LOSS_GRAD from neural_network so grad_check.py works:
    from ann.objective_functions import LOSS_FN
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from neural_network where the functions actually live
from ann.neural_network import LOSS_FN, LOSS_GRAD, _cross_entropy as cross_entropy
from ann.neural_network import _cross_entropy_grad as cross_entropy_grad
from ann.neural_network import _mse as mse, _mse_grad as mse_grad

__all__ = ["LOSS_FN", "LOSS_GRAD",
           "cross_entropy", "cross_entropy_grad", "mse", "mse_grad"]