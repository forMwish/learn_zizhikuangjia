import numpy as np

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if np.isscalar(data):
                data = np.array(data)
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not support")
        
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + 9*' ')
        return f"variable({p})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
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
            if retain_grad == False:
                for y in ys:
                    y().grad = None

    # 清楚上次计算的 grad ，避免和下次的叠加
    def cleargrad(self):
        self.grad = None