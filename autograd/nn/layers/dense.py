# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Dense(Module):
    def  __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = Parameter(np.random.randn(in_dim, out_dim))
        self.b = Parameter(np.random.randn(1, out_dim))

    def forward(self: Dense,
                X: np.ndarray) -> np.ndarray:
        result =  X @ self.W.val + self.b.val
        return result

    # also element-wise indepenent
    def backward(self: Dense,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dY_dW = np.zeros(self.W.val.shape)
        dY_db = np.zeros(self.b.val.shape)


        dY_dX = dLoss_dModule @ self.W.val.T
        if not self.W.frozen:
            dY_dW = X.T @ dLoss_dModule

        if not self.b.frozen:
            dY_db = np.sum(dLoss_dModule, axis=0, keepdims=True)

        self.W.grad = self.W.grad + dY_dW
        self.b.grad = self.b.grad + dY_db
        return dY_dX


    def parameters(self: Dense) -> List[Parameter]:
        return [self.W, self.b]