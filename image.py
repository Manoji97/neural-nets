import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

act = np.vectorize(sigmoid)

image = np.zeros((20,20))
W1 = np.random.rand(3,3)
W2 = np.random.rand(3,3)
W3 = np.random.rand(3,3)
W11 = np.random.rand(3,3)
W12 = np.random.rand(3,3)
W13 = np.random.rand(3,3)
W21 = np.random.rand(3,3)
W22 = np.random.rand(3,3)
W23 = np.random.rand(3,3)
W31 = np.random.rand(3,3)
W32 = np.random.rand(3,3)
W33 = np.random.rand(3,3)


def conv(X,w,t):
    lst = []
    for i in range(X.shape[0]):
        if X.shape[0] - i >= w.shape[0]:
            a =[]
            for j in range(X.shape[1]):
                if X.shape[1] - j >= w.shape[1]:
                    a.append(np.sum(X[i:i+w.shape[0],j:j+w.shape[1]] * w))
            lst.append(a)
    c =  np.array(lst)
    cc = c >t
    return c *cc

def pool(x,s):
    lst = []
    for i in range(x.shape[0]):
        if x.shape[0] - i >= s:
            a = []
            for j in range(x.shape[1]):
                if x.shape[1] - j >= s:
                    a.append(np.sum(x[i:i+s,j:j+s])/4)
            lst.append(a)
    return np.array(lst)


r = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

new1 =  [conv(image,W1,0.9),conv(image,W2,0.9),conv(image,W3,0.9)]
pool1 = [pool(i,2) for i in new1]
new2 = []
for i in pool1:
    new2.append(conv(i,W11,0.9))
    new2.append(conv(i,W12,0.9))
    new2.append(conv(i,W13,0.9))
pool2 = [pool(i,2) for i in new2]
new3 = []
for i in pool2:
    new3.append(conv(i,W21,0.9))
    new3.append(conv(i,W22,0.9))
    new3.append(conv(i,W23,0.9))
pool3  = [pool(i,2) for i in new3]

plt.imshow(pool1[0])




