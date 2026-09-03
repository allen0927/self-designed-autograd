# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lf import LossFunction


# TYPES DECLARED IN THIS MODULE


class BinaryCrossEntropy(LossFunction):

    def forward(self: BinaryCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        result = -1 / abs(Y_hat.shape[0]) * np.sum((Y_gt * np.log(Y_hat)) + (np.ones(Y_gt.shape) - Y_gt) * np.log(np.ones(Y_hat.shape) - Y_hat))
        return result

    def backward(self: BinaryCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        return (Y_hat - Y_gt) / (Y_hat * (1 - Y_hat) * abs(Y_hat.shape[0]))

