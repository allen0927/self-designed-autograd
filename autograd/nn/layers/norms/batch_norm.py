# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from .layer_norm import LayerNorm
from ...module import Module
from ...param import Parameter

class BatchNorm(Module):
  def __init__(self: BatchNorm,
               gamma: float = 1.0,
               beta: float = 0.0,
               epsilon: float = 1e-5) -> None:
      self.epsilon: float = epsilon
      self.gamma: float = gamma
      self.beta: float = beta
  
  def forward(self: BatchNorm,
              X: np.ndarray) -> np.ndarray:
      X_t = X.T
      E_x: np.ndarray = np.mean(X_t, axis=-1, keepdims=True)
      Var_x: np.ndarray = np.var(X_t, axis=-1, keepdims=True)
      X_normalized: np.ndarray = (X_t - E_x) / np.sqrt(Var_x + self.epsilon)
      return (self.gamma * X_normalized + self.beta).T

  def backward(self: BatchNorm,
               X: np.ndarray,
               dLoss_dModule: np.ndarray) -> np.ndarray:
      X_t = X.T
      dLoss_dModule_T = dLoss_dModule.T
      E_x: np.ndarray = np.mean(X_t, axis=-1, keepdims=True)
      Var_x: np.ndarray = np.var(X_t, axis=-1, keepdims=True)
      X_normalized: np.ndarray = (X_t - E_x) / np.sqrt(Var_x + self.epsilon)
      
      inv_std = 1.0 / np.sqrt(Var_x + self.epsilon)
      D = X_t.shape[-1]

      d_xhat = dLoss_dModule_T * self.gamma   
      sum_d_xhat = np.sum(d_xhat, axis=-1, keepdims=True)
      sum_d_xhat_xhat = np.sum(d_xhat * X_normalized, axis=-1, keepdims=True)
      dx = (1.0 / D) * inv_std * (D * d_xhat - sum_d_xhat - X_normalized * sum_d_xhat_xhat)
      return dx
  
  def parameters(self: BatchNorm) -> List[Parameter]:
     return list()