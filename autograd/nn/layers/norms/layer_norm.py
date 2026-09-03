# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ...module import Module
from ...param import Parameter



# TYPES DECLARED IN THIS MODULE

class LayerNorm(Module):
  def __init__(self: LayerNorm,
               gamma: float = 1.0,
               beta: float = 0.0,
               epsilon: float = 1e-5) -> None:
      self.epsilon: float = epsilon
      self.gamma: float = gamma
      self.beta: float = beta
  
  def forward(self: LayerNorm,
              X: np.ndarray) -> np.ndarray:
      E_x: np.ndarray = np.mean(X, axis=-1, keepdims=True)
      Var_x: np.ndarray = np.var(X, axis=-1, keepdims=True)
      X_normalized: np.ndarray = (X - E_x) / np.sqrt(Var_x + self.epsilon)
      return self.gamma * X_normalized + self.beta

  def backward(self: LayerNorm,
               X: np.ndarray,
               dLoss_dModule: np.ndarray) -> np.ndarray:
        X_t = X.T                                  
        mu = np.mean(X_t, axis=-1, keepdims=True)  
        var = np.var(X_t, axis=-1, keepdims=True)  
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        x_hat = (X_t - mu) * inv_std 

        N = X.shape[0] 
        dY_t = dLoss_dModule.T 

        d_xhat = dY_t * self.gamma 
        sum_d_xhat = np.sum(d_xhat, axis=-1, keepdims=True) 
        sum_d_xhat_xhat = np.sum(d_xhat * x_hat, axis=-1, keepdims=True) 

        dX_t = (1.0 / N) * inv_std * (N * d_xhat - sum_d_xhat - x_hat * sum_d_xhat_xhat) 
        return dX_t.T 
  
  def parameters(self: LayerNorm) -> List[Parameter]:
     return list()