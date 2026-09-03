# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Flatten(Module):
    def __init__(self: Flatten,) -> None:
        return

    # X has shape [num_examples, in_dim]
    def forward(self: Flatten,
                X: np.ndarray) -> np.ndarray:
        falttened_X = np.reshape(X, (X.shape[0], -1))
        return falttened_X

    # because our params are frozen we only need to compute and return dLoss_dX
    def backward(self: Flatten,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        dLoss_dX = dLoss_dModule.reshape(X.shape)
        return dLoss_dX

    def parameters(self: Flatten) -> List[Parameter]:
        return list()

