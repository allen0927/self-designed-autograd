# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE

DELTA: float = 1e-12
class CategoricalCrossEntropy(LossFunction):

    def forward(self: CategoricalCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        log_y_hat = np.log(Y_hat + DELTA)
        loss = -(1/Y_hat.shape[0]) * np.sum(Y_gt * log_y_hat)
        return loss

    def backward(self: CategoricalCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        dLoss_dY_hat = -(1/Y_hat.shape[0]) * (Y_gt / (Y_hat + DELTA))
        return dLoss_dY_hat
