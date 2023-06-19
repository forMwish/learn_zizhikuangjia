from ..core.function import Function
import numpy as np
from ..core.variable import Variable


def neg(x):
    return Neg()(x)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

Variable.__neg__ = neg