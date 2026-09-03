# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class MaxPool2d(Module):
    def __init__(self: MaxPool2d,
                 pool_size: Tuple[int, int],
                 stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        return


    def forward(self: MaxPool2d,
                X: np.ndarray) -> np.ndarray:
        #input batch should have shape (num examples, h, w, num channels)
        #output batch should have shape (num examples, 
        #                                 1 + (h - pool h) // stride, 
        #                                 1 + (w - pool w)// stride, 
        #                                 num channels)
        N, H, W, C = X.shape
        pool_h, pool_w = self.pool_size
        s = self.stride

        out_h = 1 + (H - pool_h) // s
        out_w = 1 + (W - pool_w) // s
        Y = np.empty((N, out_h, out_w, C), dtype=X.dtype)

        for oh in range(out_h):
            h_start_idx = oh * s
            h_end_idx   = h_start_idx + pool_h
            for ow in range(out_w):
                w_start_idx = ow * s
                w_end_idx   = w_start_idx + pool_w

                X_pool_all_channels = X[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]

                Y[:, oh, ow, :] = X_pool_all_channels.max(axis=(1, 2))
        return Y

    # because our params are frozen we only need to compute and return dLoss_dX
    def backward(self: MaxPool2d,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        N, H, W, C = X.shape
        ph, pw = self.pool_size
        s = self.stride
        # Output spatial dims (VALID pooling)
        OH = 1 + (H - ph) // s
        OW = 1 + (W - pw) // s
        dX = np.zeros_like(X, dtype=dLoss_dModule.dtype)
        for oh in range(OH):
            h_start_idx = oh * s
            h_end_idx   = h_start_idx + ph
            for ow in range(OW):
                w_start_idx = ow * s
                w_end_idx   = w_start_idx + pw

                X_pool = X[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]

                max_vals = X_pool.max(axis=(1, 2), keepdims=True)

                mask = (X_pool == max_vals)

                g = dLoss_dModule[:, oh:oh+1, ow:ow+1, :]

                dX[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :] += mask * g
        return dX

    def parameters(self: MaxPool2d) -> List[Parameter]:
        return list()
