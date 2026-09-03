# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class ReLU(Module):

    def forward(self: ReLU,
                X: np.ndarray) -> np.ndarray:
        result = np.where(X > 0, X, 0)
        return result

    # also element-wise indepenent
    def backward(self: ReLU,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y = self.forward(X)
        dY_dX = np.where(Y > 0, 1, 0)
        dLoss_dX = dLoss_dModule * dY_dX
        return dLoss_dX

    def parameters(self: ReLU) -> List[Parameter]:
        return list()
