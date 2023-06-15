from aa import *
import numpy as np


x0 = Variable(0.5)
x1 = Variable(0.1)

y0 = add(x0, x1)
y1 = square(y0)
y2 = exp(y1)
y3 = square(y2)


y3.backward()

print(x0.grad)
print(x1.grad)

