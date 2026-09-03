# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# YOUR PACKAGE IMPORTS
from autograd.nn.models.sequential import Sequential
from autograd.nn.layers.dense import Dense
from autograd.nn.layers.tanh import Tanh
from autograd.nn.layers.sigmoid import Sigmoid
from autograd.nn.optimizers.sgd import SGDOptimizer
from autograd.nn.losses.mse import MeanSquaredError
from autograd.nn.module import Module
from autograd.nn.param import Parameter  # only needed for grad_check
from autograd.nn.layers.relu import ReLU

# ----- Hyperparameters -----
in_dim: int = 100
out_dim: int = 3
hidden_dim: int = 20
lr: float = 1e-2
max_epochs: int = 200
np.random.seed(42)

# ----- Toy data -----
B = 64
X = np.random.randn(B, in_dim)
# targets in {0,1} so Sigmoid + MSE is reasonable for a quick smoke test
Y_gt = (np.random.rand(B, out_dim) > 0.5).astype(float)

# ----- Build model -----
m: Sequential = Sequential()
m.add(Dense(in_dim, hidden_dim))
m.add(ReLU())                 # <-- ReLU here
m.add(Dense(hidden_dim, out_dim))
m.add(Sigmoid())

# ----- Optimizer & loss -----
optim: SGDOptimizer = SGDOptimizer(m.parameters(), lr)
loss_func: MeanSquaredError = MeanSquaredError()

# ----- (Optional) quick gradient check helper -----
def grad_check(X: np.ndarray,
               Y_gt: np.ndarray,
               m: Module,
               ef: MeanSquaredError,
               epsilon: float = 1e-4,
               delta: float = 1e-6) -> None:
    # Call AFTER a backward pass so .grad is populated
    params: List[Parameter] = m.parameters()
    num_grads: List[np.ndarray] = [np.zeros_like(P.val) for P in params]
    sym_grads: List[np.ndarray] = [np.array(P.grad, copy=True) for P in params]

    for P, N in zip(params, num_grads):
        it = np.nditer(P.val, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            v = P.val[idx]
            P.val[idx] = v + epsilon
            f_pos = ef.forward(m.forward(X), Y_gt)
            P.val[idx] = v - epsilon
            f_neg = ef.forward(m.forward(X), Y_gt)
            P.val[idx] = v
            N[idx] = (f_pos - f_neg) / (2.0 * epsilon)
            it.iternext()

    ratios = np.array([
        np.linalg.norm(sg - ng) / (np.linalg.norm(sg + ng) + 1e-12)
        for sg, ng in zip(sym_grads, num_grads)
    ], dtype=float)
    if np.any(ratios > delta) or np.isnan(ratios).any():
        raise RuntimeError(f"Grad check FAILED: delta={delta}, ratios={ratios}")
    else:
        print("Grad check passed. max ratio =", float(np.max(ratios)))

# ----- Train -----
losses: List[float] = []
for epoch in range(max_epochs):
    # zero grads
    if hasattr(optim, "reset"):
        optim.reset()
    else:
        for p in m.parameters():
            p.grad = np.zeros_like(p.val)

    # forward
    Y_hat = m.forward(X)

    # loss
    loss_val = loss_func.forward(Y_hat, Y_gt)
    losses.append(loss_val)

    # backward
    dL_dY = loss_func.backward(Y_hat, Y_gt)
    m.backward(X, dL_dY)

    # (optional) one-time grad check (expensive)
    # if epoch == 0:
    #     grad_check(X, Y_gt, m, loss_func)

    # step
    if hasattr(optim, "step"):
        optim.step()
    else:
        for p in m.parameters():
            optim.step_parameter(p)

    if (epoch + 1) % 25 == 0:
        print(f"epoch {epoch+1:3d} | loss {loss_val:.6f}")

# ----- Plot or save -----
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True)
if matplotlib.get_backend().lower().endswith("agg"):
    plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
    print("Saved plot to loss_curve.png (Agg backend)")
else:
    plt.show()
