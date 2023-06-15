from aa import *
import numpy as np


# A = Square()
# B = Exp()
# C = Square()

# x = Variable(0.5)

# y0 = A(x)
# y1 = B(y0)
# y2 = C(y1)

# y2.backward()

# print(x.grad)

x0 = Variable(0.1)
x1 = Variable(1)

y = add(x0, x1)
print(y.data)
print(type(y.data))