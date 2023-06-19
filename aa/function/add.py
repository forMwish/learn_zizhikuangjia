from .function import Function
import numpy as np
from ..variable import Variable


def add(x0, x1):
    return Add()(x0, x1)

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy

Variable.__add__ = add
Variable.__radd__ = add