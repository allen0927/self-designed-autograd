# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Union, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter
from .cells.rnn_cell import RNNCell


# TYPES DECLARED IN THIS MODULE


class RNN(Module):
    def __init__(self,
                 cell: RNNCell,
                 return_sequences: bool = False,
                 return_states: bool = False,
                 backprop_through_time_limit: int | None = None) -> None:
        super().__init__()
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.backprop_through_time_limit = backprop_through_time_limit

    def init_states(self: RNN, batch_size: int) -> np.ndarray:
        return self.cell.init_states(batch_size)

    def forward(self: RNN,
                X: np.ndarray,
                states_init: np.ndarray = None
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_size, seq_len = X.shape[0], X.shape[1]
        H = self.init_states(batch_size) if states_init is None else states_init
        all_states, all_predictions = [], []
        all_states.append(H)
        for t in range(seq_len):
            X_t = X[:, t, ...]
            H, A_t = self.cell.forward(H, X_t)
            all_states.append(H)
            all_predictions.append(A_t)
        if self.return_states:
            if self.return_sequences:
                returned_states = np.stack(all_states, axis=1)
            else:
                returned_states = all_states[-1]
            
            returned_pred = (
                np.stack(all_predictions, axis=1)
                if self.return_sequences
                else all_predictions[-1]
            )
            return returned_states, returned_pred
        else:
            returned_pred = np.stack(all_predictions, axis=1) if self.return_sequences else all_predictions[-1]
            return returned_pred

    def backward(self: RNN,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray,
                 states_init: np.ndarray = None) -> np.ndarray:
        batch_size, seq_len = X.shape[0], X.shape[1]
        H = self.cell.init_states(batch_size) if states_init is None else states_init
        H0 = H.copy()
        all_states, all_predictions = [], []
        all_states.append(H0)
        for t in range(seq_len):
            X_t = X[:, t, ...]
            H, A_t = self.cell.forward(H, X_t)
            all_states.append(H)
            all_predictions.append(A_t)

        hidden_dim = all_states[0].shape[1]
        grad_h_next = np.zeros((batch_size, hidden_dim))
        dLoss_dX: List[np.ndarray] = [None] * seq_len

        K = self.backprop_through_time_limit
        t_min = 0 if (K is None) else max(0, seq_len - K)

        for t in range(seq_len - 1, -1, -1):
            X_t = X[:, t, ...]
            H_t_minus_1 = all_states[t] #CHECK

            if self.return_sequences:
                dA_t = dLoss_dModule[:, t, ...]
            else:
                dA_t = dLoss_dModule if t == seq_len - 1 else np.zeros_like(all_predictions[-1])

            dH_prev, dX_t = self.cell.backward(H_t_minus_1, X_t, dA_t, grad_h_next)
            dLoss_dX[t] = dX_t

            grad_h_next = dH_prev
            if (K is not None) and (t <= t_min):
                grad_h_next = np.zeros_like(grad_h_next)

        dLoss_dX = np.stack(dLoss_dX, axis=1)
        return dLoss_dX

    def parameters(self: RNN) -> List[Parameter]:
        return self.cell.parameters()