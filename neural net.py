import random
import math


class neuron:
    def __init__(self,ni):
        self.numInputs = ni
        self.Inputs = []
        self.bias = random.randint(-100,100)/100
        self.weights = []
        self.errorgrad = 0
        self.Output = 0
        for i in range(self.numInputs):
            self.weights.append(random.randint(-100,100)/100)

class layer:
    def __init__(self,num_neu,num_inputs):
        self.numNeurons = num_neu
        self.numInputs = num_inputs
        self.neuronList = []

        for i in range(self.numNeurons):
            self.neuronList.append(neuron(self.numInputs))


class neural_network:
    def __init__(self,num_inputs,num_outputs,num_hidden,num_neuron_layer,alpha):
        self.numInputs = num_inputs
        self.numOutputs = num_outputs
        self.numNeuronLayer = num_neuron_layer
        self.numHidden = num_hidden
        self.alpha = alpha
        self.layerList = []


    def neural_net(self):
        if self.numHidden > 0:
            self.layerList.append(layer(self.numNeuronLayer,self.numInputs))                    # input layer
            for i in range(self.numHidden -1):
                self.layerList.append(layer(self.numNeuronLayer,self.numNeuronLayer))           # hidden layers
            self.layerList.append(layer(self.numOutputs,self.numNeuronLayer))
        else:
            self.layerList.append(layer(self.numOutputs,self.numInputs))                        # output layer

    def forward_pass(self,inputs):
        Inputs = inputs
        Outputs = []
        for i in range(self.numHidden+1):
            if i > 0:
                Inputs = Outputs.copy()
            Outputs.clear()
            for j in range(self.layerList[i].numNeurons):
                x = 0
                for k in range(self.layerList[i].neuronList[j].numInputs):
                    self.layerList[i].neuronList[j].Inputs.append(Inputs[k])
                    x += self.layerList[i].neuronList[j].weights[k] * Inputs[k]
                x += self.layerList[i].neuronList[j].bias
                if i == self.numHidden: x = self.activation(x)
                else: x = self.activation(x)
                self.layerList[i].neuronList[j].Output = x
                Outputs.append(x)
        return Outputs


    def back_pass(self,out,tar):
        for i in range(self.numHidden,-1,-1):
            for j in range(self.layerList[i].numNeurons):
                if i == self.numHidden:
                    err = out[j] - tar[j]
                    self.layerList[i].neuronList[j].errorgrad = err * self.layerList[i].neuronList[j].Output * (1 - self.layerList[i].neuronList[j].Output)
                else:
                    self.layerList[i].neuronList[j].errorgrad = self.layerList[i].neuronList[j].Output * (1 - self.layerList[i].neuronList[j].Output)
                    grad_sum = 0
                    for p in range(self.layerList[i+1].numNeurons):
                        grad_sum += self.layerList[i+1].neuronList[p].errorgrad * self.layerList[i+1].neuronList[p].weights[j]
                    self.layerList[i].neuronList[j].errorgrad *= grad_sum

                for k in range(self.layerList[i].neuronList[j].numInputs):
                    self.layerList[i].neuronList[j].weights[k] -= self.alpha * self.layerList[i].neuronList[j].Inputs[k] * self.layerList[i].neuronList[j].errorgrad
                self.layerList[i].neuronList[j].bias -= self.alpha * self.layerList[i].neuronList[j].errorgrad

                self.layerList[i].neuronList[j].Inputs.clear()          # clear the inputs 



    

    def train(self,inputs,target):
        output = self.forward_pass(inputs)
        self.back_pass(output,target)
        return output

    def tanh(self,val):
        x = 2 * self.activation(2*val) - 1
        return x
                
    def activation(self,val):
        x = math.exp(-val)
        return 1/(1+x)

    def relu(self,val):
        if val < 0 : return 0
        else: return val
    def softmax(self,val):
        if val < 0: return 0
        else: return 1



n = neural_network(2,1,1,3,0.8)
n.neural_net()


for i in range(10000):
    error = 0
    o = n.train([1,1],[0])
    error += math.pow(o[0] - 0,2)
    o = n.train([1,0],[1])
    error += math.pow(o[0] - 1,2)
    o = n.train([0,1],[1])
    error += math.pow(o[0] - 1,2)
    o = n.train([0,0],[0])
    error += math.pow(o[0] - 0,2)

    if i % 100 == 0:
        print(error)


print(n.forward_pass([1,1]))
print(n.forward_pass([1,0]))     
print(n.forward_pass([0,1]))
print(n.forward_pass([0,0]))























            
