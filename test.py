from aa.core import Variable
from aa.function import *
from aa.utils import no_grad, plot_dot_graph
import numpy as np
import os

## test 0
# x = Variable(0.5)
# y0 = square(x)
# y2 = exp(y0)
# y3 = square(y0)

# y4 = y2 + y3

# y4.backward()

# print(x.grad)

# with no_grad():
#     x = Variable(0.5)
#     y0 = square(x)
#     y2 = exp(y0)
#     y3 = square(y0)

#     y4 = y2 + y3
#     print(x.grad)

# test 1
def sphere(x, y):
    z = x**2 + y**2
    return z

def matyas(x, y):
    z = 0.26*(x**2 + y**2) - 0.48*x*y
    return z

def goldstein(x, y):
    z = (1 + (x+y+1)**2 * (19-14*x +3*x**2 -14*y +6*x*y + 3*y**2)) *\
        (30 +(2*x-3*y)**2 *(18-32*x +12*x**2 + 48*y -36*x*y + 27*y**2))
    return z

x = Variable(1.0)
y = Variable(1.0)
z = goldstein(x, y) 
z.backward()

x.name = "x"
y.name = "y"

z.name = "out"

print(x.grad)
print(y.grad)

plot_dot_graph(z, verbose=False, out_path="./output")







