import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

training_data = []

for i in range(5):
    training_data.append(np.zeros((10,10)))

os.chdir("C:\\Users\\u691987\\Desktop\\New folder")

with open("small_input","wb") as file:
    pickle.dump(training_data,file)
    
Input = training_data
Y = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])

# weights initialization
K1 = []
B1 = []
for i in range(6):
    K1.append(np.random.rand(3,3)-1/2)
    B1.append(np.random.rand(1,1)-1/2)    
    
W = np.random.rand(4,96)-1/2
B = np.random.rand(4,1)-1/2

# functions
def sig(x):
    return 1/(1+np.exp(-x))
sigmoid = np.vectorize(sig)

def Conv(X,W):
    lst = []
    for i in range(X.shape[0]):
        if X.shape[0] - i >= W.shape[0]:
            a =[]
            for j in range(X.shape[1]):
                if X.shape[1] - j >= W.shape[1]:
                    a.append(np.sum(X[i:i+W.shape[0],j:j+W.shape[1]] * W))
            lst.append(a)
    return np.array(lst)

def Pool(X):
    lst = []
    for i in range(0,X.shape[0],2):
        if X.shape[0] - i >= 2:
            a = []
            for j in range(0,X.shape[1],2):
                if X.shape[1] - j >= 2:
                    a.append(np.sum(X[i:i+2,j:j+2])/4)
            lst.append(a)
    return np.array(lst)

def dePool(X):
    a=  np.zeros((2*X.shape[0],2*X.shape[1]))
    for i in range(2*X.shape[0]):
        for j in range(2*X.shape[1]):
            a[i,j] = X[int(i/2),int(j/2)]
    return a
            

    
def flat(L):
    a = np.array([[]])
    for i in L:
        i.flatten()
        a = np.concatenate((a,i),axis = None)
    return a
        
def Cost(o,i):
    j = np.sum(np.square(o - Y[i].reshape(4,1)))/8
    return j


