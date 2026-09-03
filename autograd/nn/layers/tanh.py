# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Tanh(Module):

    def forward(self: Tanh,
                X: np.ndarray) -> np.ndarray:
        result = np.tanh(X)
        return result

    # also element-wise indepenent
    def backward(self: Tanh,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        Y = self.forward(X)
        dY_dX = 1 - Y**2
        dLoss_dX = dLoss_dModule * dY_dX
        return dLoss_dX

    def parameters(self: Tanh) -> List[Parameter]:
        return list()
