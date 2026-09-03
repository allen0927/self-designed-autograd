# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class TimeDistributed(Module):
    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, X: np.ndarray) -> np.ndarray:
        batch_size, seq_len = X.shape[0], X.shape[1]
        X_flat = X.reshape(batch_size * seq_len, *X.shape[2:])
        Y_flat = self.module.forward(X_flat)
        Y = Y_flat.reshape(batch_size, seq_len, *Y_flat.shape[1:])
        return Y

    def backward(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        batch_size, seq_len = X.shape[0], X.shape[1]
        X_flat = X.reshape(batch_size * seq_len, *X.shape[2:])
        dY_flat = dLoss_dModule.reshape(batch_size * seq_len, *dLoss_dModule.shape[2:])
        dX_flat = self.module.backward(X_flat, dY_flat)
        dX = dX_flat.reshape(batch_size, seq_len, *X.shape[2:])
        return dX

    def parameters(self: TimeDistributed) -> List[Parameter]:
        return self.module.parameters()
