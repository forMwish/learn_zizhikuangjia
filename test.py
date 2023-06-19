from aa.core import Variable
from aa.function import *
from aa.other import no_grad
import numpy as np


x = Variable(0.5)
y0 = square(x)
y2 = exp(y0)
y3 = square(y0)

y4 = y2 + y3

y4.backward()

print(x.grad)

with no_grad():
    x = Variable(0.5)
    y0 = square(x)
    y2 = exp(y0)
    y3 = square(y0)

    y4 = y2 + y3
    print(x.grad)

# a = Variable(np.array(3.0))
# b = Variable(np.array(2.0))
# y = a*np.array([3])
# print(y)







