# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..optimizer import Optimizer
from ..param import Parameter



class SGDOptimizer(Optimizer):
    def step_parameter(self: SGDOptimizer, P: Parameter) -> None:
        P.val -= self.lr * P.grad
