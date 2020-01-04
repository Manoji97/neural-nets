import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

activation = np.vectorize(sigmoid)

X = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

alpha = 0.8  
W1 = np.random.rand(2,3)
B1 = np.ones((4,3))
W2 = np.random.rand(3,1)
B2 = np.ones((4,1))

for i in range(1000):
    #forward pass
    net1 = X.dot(W1) + B1
    out1 = activation(net1)
    net2 = out1.dot(W2) + B2
    out = activation(net2)
    
    #back pass
    J = np.sum(np.square(out-Y))/8
    Eg = (out-Y) *(out)*(1- out)
    Eg1 = (out1) *(1-out1) * (Eg.dot(W2.transpose()))
    
    W2 = W2 - alpha*(out1.transpose().dot(Eg))
    B2 = B2 - alpha*(np.sum(Eg))
    W1 = W1 - alpha* (X.transpose().dot(Eg1))
    B1 = B1 - alpha* (Eg1.sum(axis = 0))
    if i %100 == 0:
        print(J)
    






X = np.array([[1,1]])
net1 = X.dot(W1) + B1
out1 = activation(net1)
net2 = out1.dot(W2) + B2
out = activation(net2)





    

                
        