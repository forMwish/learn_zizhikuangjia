from ..core.function import Function
import numpy as np
from ..core.variable import Variable


def pow(x0, x1):
    return Pow()(x0, x1)

class Pow(Function):
    def forward(self, x0, x1):
        return x0**x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1*x0**(x1 -1)

Variable.__pow__ = pow