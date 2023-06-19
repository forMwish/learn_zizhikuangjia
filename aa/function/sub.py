from .function import Function
import numpy as np
from ..variable import Variable


def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    return Sub()(x1, x0)

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy

Variable.__sub__ = sub
Variable.__rsub__ = rsub