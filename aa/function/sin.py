from ..core.function import Function
import numpy as np
from ..core.variable import Variable


def sin(x):
    return Sin()(x)

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy*np.cos(self.inputs[0].data)
               