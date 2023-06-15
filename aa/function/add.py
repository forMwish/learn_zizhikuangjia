from .function import Function
import numpy as np

def add(xs):
    return Add()(xs)

class Add(Function):
    def forward(self, xs):
        out = np.zeros_like(xs[0])
        for x in xs:
            out += x
        return [out]