import random
import math


class neuron:
    def __init__(self,nI):
        self.numInputs = nI
        self.inputs = []
        self.bias = random.randint(-100,100)/100
        self.weights = []
        self.output = 0
        self.error = 0
        for i in range(self.numInputs):
            self.weights.append(random.randint(-100,100)/100)

    def train(self,inputs,target,switch):
        self.inputs = inputs
        for i in range(self.numInputs):
            self.output +=self.inputs[i] * self.weights[i]
        self.output += self.bias
        self.output = self.softmax(self.output)

        if switch == 1:
            self.updateWeights(target,self.output)
            
        return self.output

    def updateWeights(self,t,y):
        err = t-y
        for i in range(self.numInputs):
            self.weights[i] += err*self.inputs[i]
        self.bias += err

    def sigmoid(self,val):
        x = float(math.exp(val))
        return x/(1+x)
    
    def softmax(self,val):
        if val > 0:
            return 1
        else:
            return 0
        
 

        
n = neuron(2)
print(n.weights)
for i in range(1000):
    error = 0
    res = n.train([1,1],1,1)
    error += math.pow((res - 1),2)
    res = n.train([1,0],0,1)
    error += math.pow((res - 0),2)
    res = n.train([0,1],0,1)
    error += math.pow((res - 0),2)
    res = n.train([0,0],0,1)
    error += math.pow((res - 0),2)

    if (i% 100) == 0:
        print(error)
        print(n.weights,n.bias)

    
    
            
