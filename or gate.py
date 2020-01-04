import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def costfunction(O,T):
    J = np.sum(np.square(O - T))/4
    errorgrad = O *(1-O)*(O-T)
    return J,errorgrad
   
activation = np.vectorize(sigmoid)

X = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])

W = np.random.rand(2,1)

alpha = 0.8
Y = np.array([[1],
              [1],
              [1],
              [0]])
a = []
b = []
B = np.ones(Y.shape)


for i in range(200):
    net = (X.dot(W)) + B
    out = activation(net)
    J,Eg = costfunction(out,Y)
    W = W - (alpha* X.transpose().dot(Eg))
    B = B - (alpha* np.sum(Eg))
    a.append(J)
    b.append(W[0,0])

plt.plot(a)


print(X.transpose().dot(Eg))
    




    



