# sequential.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

from ..module import Module
from ..param import Parameter

class Sequential(Module):
    def __init__(self, modules: Optional[List[Module]] = None) -> None:
        super().__init__()
        self.modules: List[Module] = list(modules) if modules is not None else []
        self._inputs: List[np.ndarray] = []


    def add(self, module: Module) -> None:
        self.modules.append(module)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._inputs = [] 
        out = X
        for mod in self.modules:
            self._inputs.append(out)
            out = mod.forward(out)
        return out

    def backward(self, X: np.ndarray, dLoss_dOutput: np.ndarray) -> np.ndarray:
        grad = dLoss_dOutput
        for i in reversed(range(len(self.modules))):
            grad = self.modules[i].backward(self._inputs[i], grad)
        return grad

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for mod in self.modules:
            params.extend(mod.parameters())
        return params
