import numpy as np

class Variable:
    def __init__(self, x):
        if x is not None:
            if np.isscalar(x):
                x = np.array(x)
            if not isinstance(x, np.ndarray):
                raise TypeError(f"{type(x)} is not support")
        
        self.data = x
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            func = funcs.pop()

            xs = func.inputs
            ys = func.outputs
            gys = [y.grad for y in ys]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(xs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)