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
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(func):
            if func not in seen_set:
                seen_set.add(func)
                funcs.append(func)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            func = funcs.pop()

            xs = func.inputs
            ys = func.outputs
            gys = [y().grad for y in ys]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(xs, gxs):
                if x.grad == None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # 注意不能使用 x.grad += gx; 因为这样会导致在已有的 x.grad 对象上进行加法；而这个对象是上次的 gx(加法的 gx 未拷贝) 

                if x.creator is not None:
                    add_func(x.creator)

    # 清楚上次计算的 grad ，避免和下次的叠加
    def cleargrad(self):
        self.grad = None