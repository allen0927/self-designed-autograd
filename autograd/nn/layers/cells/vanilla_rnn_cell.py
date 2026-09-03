# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ...module import Module
from .rnn_cell import RNNCell
from ...param import Parameter
from ..dense import Dense
from ..tanh import Tanh
from ..sigmoid import Sigmoid

# TYPES DECLARED IN THIS MODULE


class VanillaRNNCell(RNNCell):
    def __init__(self: RNNCell,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 hidden_activation: Module = None,
                 output_activation: Module = None) -> None:
        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.out_dim: int = out_dim
        self.hidden_activation: Module = hidden_activation if hidden_activation is not None else Tanh()
        self.out_activation: Module = output_activation if output_activation is not None else Sigmoid()

        self.hidden_dense: Module = Dense(in_dim + hidden_dim, hidden_dim)
        self.out_dense: Module = Dense(hidden_dim, out_dim)

    def init_states(self: RNNCell,
                batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_dim), dtype=float)

    def forward(self: VanillaRNNCell,
                H_t_minus_1: np.ndarray,
                X_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #X_t shape: (batch_size, in_dim)
        #H_t_minus_1 shape: (batch_size, hidden_dim)
        concatenated = np.concatenate((X_t, H_t_minus_1), axis=1)
        Z_t = self.hidden_dense.forward(concatenated)
        H_t = self.hidden_activation.forward(Z_t)
        R_t = self.out_dense.forward(H_t)
        A_t= self.out_activation.forward(R_t)

        return H_t, A_t
    
    # also element-wise indepenent
    def backward(self: VanillaRNNCell,
                H_t_minus_1: np.ndarray,
                X_t: np.ndarray,
                dLoss_dModule_t: np.ndarray,
                dLoss_dStates_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        used_A_t:bool = dLoss_dModule_t is not None
        used_H_t:bool = dLoss_dStates_t is not None

        Xt_H_t_minus_1 = np.concatenate((X_t, H_t_minus_1), axis=1)
        Z_t = self.hidden_dense.forward(Xt_H_t_minus_1)
        H_t = self.hidden_activation.forward(Z_t)
        R_t = self.out_dense.forward(H_t)

        dLoss_dHt = np.zeros_like(H_t)

        if used_A_t:
            dLoss_dRt = self.out_activation.backward(R_t, dLoss_dModule_t)
            dLoss_dHt += self.out_dense.backward(H_t, dLoss_dRt)

        if used_H_t:
            dLoss_dHt += dLoss_dStates_t
        dLoss_dZt = self.hidden_activation.backward(Z_t, dLoss_dHt)
        dLoss_dconcat = self.hidden_dense.backward(Xt_H_t_minus_1, dLoss_dZt)
        
        dloss_dX_t = dLoss_dconcat[:, :self.in_dim]
        dloss_dH_t_minus_1 = dLoss_dconcat[:, self.in_dim:]

        return dloss_dH_t_minus_1, dloss_dX_t

    def parameters(self: VanillaRNNCell) -> List[Parameter]:
        return self.hidden_dense.parameters() + self.hidden_activation.parameters()\
                 + self.out_dense.parameters() + self.out_activation.parameters()

