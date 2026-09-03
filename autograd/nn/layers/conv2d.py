# SYSTEM IMPORTS
from __future__ import annotations
from typing import List,Union,Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


class Conv2d(Module):

    # the formula of sigmoid is 1/(1 + e^-x)
    def __init__(self: Conv2d,
                num_kernels: int,
                num_channels: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: int = 1,
                padding: str = "valid") -> np.ndarray:
        #self.W have size (kernel height, kernel width, num channels, num kernels)
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        W_data = 0.01 * np.random.randn(kh, kw, num_channels, num_kernels)
        b_data = np.zeros((num_kernels,))
        self.W = Parameter(W_data)
        self.b = Parameter(b_data)

        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.kernel_size =  (kh, kw)
        self.padding = padding.lower()
        self.stride = stride

        self.pad_amounts: Tuple[int, int] = self.get_pad_amounts()

        return
    

    def get_pad_amounts(self: Conv2d) -> tuple[int,int]:
        if self.padding == "valid":
            return (0, 0)
        kh, kw = self.kernel_size
        return (kh // 2, kw // 2)
    

    def pad_imgs(self: Conv2d, X: np.ndarray) -> np.ndarray:
        pad_h, pad_w = self.pad_amounts
        if pad_h == 0 and pad_w == 0:
            return X
        return np.pad(
            X,
            pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    

    def get_out_shape(self: Conv2d, X_padded) -> tuple[int,int,int,int]:

        if isinstance(X_padded, tuple):
            N, H_in, W_in, C_in = X_padded
        else:
            N, H_in, W_in, C_in = X_padded.shape

        kh, kw = self.kernel_size
        s = self.stride
        pad_h, pad_w = self.get_pad_amounts()

        H_out = (H_in - kh + 2*pad_h) // s + 1
        W_out = (W_in - kw + 2*pad_w) // s + 1
        return (N, H_out, W_out, self.num_kernels)


    def forward(self: Conv2d, X: np.ndarray) -> np.ndarray:
        Xp = self.pad_imgs(X)

        N, H_out, W_out, C_out = self.get_out_shape(X.shape)
        Y = np.empty((N, H_out, W_out, C_out), dtype=X.dtype)

        kh, kw = self.kernel_size
        s = self.stride
        W = self.W.val
        b = self.b.val 

        for oh in range(H_out):
            h_start = oh * s
            h_end = h_start + kh
            for ow in range(W_out):
                w_start = ow * s
                w_end = w_start + kw
                X_win = Xp[:, h_start:h_end, w_start:w_end, :]

                Y[:, oh, ow, :] = X_win.reshape(N, -1).dot(W.reshape(-1, C_out)) + b

        return Y
    

    def backward(self: Conv2d, 
                 X: np.ndarray, 
                 dLoss_dModule: np.ndarray) -> np.ndarray:

        N, H_out, W_out, C_out = self.get_out_shape(X.shape)

        Xp = self.pad_imgs(X) 
        Hp, Wp, Cin = Xp.shape[1], Xp.shape[2], Xp.shape[3]
        kh, kw = self.kernel_size
        s = self.stride
        W = self.W.val

        dloss_dx  = np.zeros_like(Xp)
        dloss_dW = np.zeros_like(W)
        dloss_db = dLoss_dModule.sum(axis=(0,1,2))

        W_rs = W.reshape(-1, C_out)
        for oh in range(H_out):
            h_start = oh * s
            h_end = h_start + kh
            for ow in range(W_out):
                w_start = ow * s
                w_end = w_start + kw

                X_win = Xp[:, h_start:h_end, w_start:w_end, :]

                dY_win= dLoss_dModule[:, oh, ow, :]

                dloss_dx[:, h_start:h_end, w_start:w_end, :] += dY_win.dot(W_rs.T).reshape(N, kh, kw, Cin)

                dloss_dW += X_win.reshape(N, -1).T.dot(dY_win).reshape(kh, kw, Cin, C_out)


        self.W.grad = dloss_dW
        self.b.grad = dloss_db

        if self.padding == "same":
            return dloss_dx[:, self.pad_amounts[0]:-self.pad_amounts[0], self.pad_amounts[1]:-self.pad_amounts[1], :]
        return dloss_dx
    def parameters(self: Conv2d) -> List[Parameter]:
        return [self.W, self.b] 
