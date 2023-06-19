from ..core.function import Function
import numpy as np
from ..core.variable import Variable


def mul(x0, x1):
    return Mul()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        return self.inputs[1].data*gy, self.inputs[0].data*gy

Variable.__mul__ = mul
Variable.__rmul__ = mul
