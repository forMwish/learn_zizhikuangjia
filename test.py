from aa import *
import numpy as np


x = Variable(0.5)
y0 = square(x)
y2 = exp(y0)
y3 = square(y0)

y4 = add(y2, y3)

y4.backward()

print(x.grad)

with no_grad():
    x = Variable(0.5)
    y0 = square(x)
    y2 = exp(y0)
    y3 = square(y0)

    y4 = add(y2, y3)
    print(x.grad)




