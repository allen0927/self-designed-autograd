# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE


class Sigmoid(Module):

    def forward(self: Sigmoid,
                X: np.ndarray) -> np.ndarray:
        result = 1 / (1 + np.exp(-X))
        return result

    # also element-wise indepenent
    def backward(self: Sigmoid,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dY_dX = self.forward(X) * (1 - self.forward(X))
        dLoss_dX = dLoss_dModule * dY_dX
        return dLoss_dX

    def parameters(self: Sigmoid) -> List[Parameter]:
        return list()
