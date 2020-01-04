import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

os.chdir("C:\\Users\\u691987\\Desktop\\New folder")


with open("Input data","rb") as file:
    Training_data = pickle.load(file)

Y = np.array([[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]])
    

# Functions
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

    
def Pool22(X):
    lst = []
    for i in range(0,X.shape[0],2):
        if X.shape[0] - i >= 2:
            a = []
            for j in range(0,X.shape[1],2):
                if X.shape[1] - j >= 2:
                    a.append(np.sum(X[i:i+2,j:j+2])/4)
            lst.append(a)
    return np.array(lst)
  

def depool(X):
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
    




# Parameter Initialization
K1 = []
B1 = []
for i in range(6):
    K1.append(np.random.rand(5,5)-1/2)
    B1.append(np.random.rand(1,1)-1/2)

K2 = []
B2 = []
for i in range(6):
    temp = []
    for j in range(12):
        temp.append(np.random.rand(5,5)-1/2)
    K2.append(temp)
for i in range(12):
    B2.append(np.random.rand(1,1)-1/2)

W = np.random.rand(4,192)-1/2
B = np.random.rand(4,1)-1/2

# Forward Pass
conv1= []
for i in range(6):
    o = Conv(Training_data[0],K1[i])
    o = o + B1[i]
    o = sigmoid(o)
    conv1.append(o)

pool1 = [Pool22(i) for i in conv1]
conv2 = []

for i in range(6):
    temp = []
    for j in range(12):
        o = Conv(pool1[i],K2[i][j])
        temp.append(o)
    conv2.append(temp)

c = []
for i in range(12):
    s = 0
    for j in range(6):
        s = s + conv2[j][i]
    c.append(s)
conv2 = []
for i in range(12):
    o = o + B2[i] 
    o = sigmoid(o)
    conv2.append(o)

pool2 = [Pool22(i) for i in conv2]

Flayer = flat(pool2).reshape(192,1)

net = W.dot(Flayer) + B
out = sigmoid(net)

J = Cost(out,0)

Eg = (out-Y[0].reshape(4,1)) *(out)*(1- out)      # for W and B





   
  

  
    
    
    
    
    
    
    
    
    
    
    
    
    