# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np
from enum import Enum


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter

# Way of skip connection
class SkipConnectionType(Enum):
    ADD = 0
    CONCATENATE = 1

# TYPES DECLARED IN THIS MODULE


class SkipConnection(Module):
    def __init__(self: SkipConnection, 
                 module: Module,
                 mode: SkipConnectionType = SkipConnectionType.ADD):
        self.module: Module = module
        self.mode: SkipConnectionType = mode
        self.forward_out_dim = None
        

    def forward(self: SkipConnection,
                X: np.ndarray) -> np.ndarray:
        F_x = self.module.forward(X)
        if self.mode == SkipConnectionType.ADD:
            return F_x + X
        
        elif self.mode == SkipConnectionType.CONCATENATE:
            self.forward_out_dim = F_x.shape[-1]
            return np.concatenate((F_x, X), axis=-1)
        
        raise ValueError("Invalid SkipConnection mode.")

    # also element-wise indepenent
    # derivative of tanh: (1- tanh(x)**2)
    def backward(self: SkipConnection,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        if self.mode == SkipConnectionType.ADD:
            return dLoss_dModule + self.module.backward(X, dLoss_dModule)

        elif self.mode == SkipConnectionType.CONCATENATE:
            F_x_shape = self.module.forward(X).shape[-1]
            dL_dFx = dLoss_dModule[:, :F_x_shape]
            dL_dx = dLoss_dModule[:, F_x_shape:]
            return dL_dx + self.module.backward(X, dL_dFx) 
        
        else:
            raise ValueError("UNKNOWN Residual Connection Type")

    def parameters(self: SkipConnection) -> List[Parameter]:
        return self.module.parameters()

