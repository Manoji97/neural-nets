import random
import os
import math
import numpy as np
import matplotlib.pylab as plt

def f1(x):
    return (2*x*x*np.cos(x)) - (5*x)

f = np.vectorize(f1)

x = np.arange(-5,5.5,0.5)
plt.plot(x,f(x))
plt.show()

    
rate = 0.01

def slope(x):
    return (4*x*np.cos(x)) - 5 - (2*x*x*np.sin(x))

x_o = 0
for i in range(20):
    x_n = x_o - (rate*slope(x_o))
    x_o = x_n
    print(x_o)
    plt.plot(x,f(x),x_o,f(x_o),'bo')
    plt.show()




