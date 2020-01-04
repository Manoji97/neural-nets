import random
import math

class neuron:
    def __init__(self,num_ip):
        self.Num_ip = num_ip
        self.bias = random.random()
        self.eg = 0
        self.ip_lst = []
        self.wt_lst = []
        self.out = 0
        for i in range(num_ip):
            self.wt_lst.append(random.random())

class layer:
    def __init__(self,num_neu,num_ip_neu):
        self.num_neu = num_neu
        self.layer = []
        for i in range(num_neu):
            self.layer.append(neuron(num_ip_neu))


class neural_net:
    def __init__(self,n_i,n_o,n_h,n_p_h,a):
        self.num_ip = n_i
        self.num_op = n_o
        self.num_hidden = n_h
        self.num_per_hidden = n_p_h
        self.learning_rate = a
        self.layer_lst = []
        if (self.num_hidden > 0):
            self.layer_lst.append(layer(self.num_per_hidden,self.num_ip))
            for i in range(self.num_hidden-1):
                self.layer_lst.append(layer(self.num_per_hidden,self.num_per_hidden))
            self.layer_lst.append(layer(self.num_op,self.num_per_hidden))
        else:
            self.layer_lst.append(layer(self.num_op,self.num_ip))


    def train(inputs,target):
        ipn_lst = inputs
        opn_lst = []

        for i in range(self.num_hidden + 1):
            if (i > 0):
                ipn_lst = opn_lst
            opn_lst.clear()

            for j in range(layer_lst[i].num_neu):
                n = 0
                self.layer_lst[i].layer[j].ip_lst.clear()
                for k in range(self.layer_lst[i].layer[j].Num_ip):
                    self.layer_lst[i].layer[j].ip_lst.append(ipn_lst[k])
                    n += self.layer_lst[i].layer[j].wt_lst[k] * ipn_lst[k]
                n -= self.layer_lst[i].layer[j].bias
                self.layer_lst[i].layer[j].out = func(n)
                opn_lst.append(self.layer_lst[i].layer[j].out)
        update_weights(opn_lst,target)
        return opn_lst


    def update_weights(out,tar):
        err = 0
        for i in range(self.num_hidden,-1,-1):
            for j in range(self.num_per_hidden):
                if i == self.num_hidden:
                    err = tar[j] - out[j]
                    self.layer_lst[i].layer[j].eg = out[j] * (1 - out[j]) * err
                else:
                    self.layer_lst[i].layer[j].eg = self.layer_lst[i].layer[j].out * (1 - self.layer_lst[i].layer[j].out)
                    sum_grad = 0
                    for p in range(layer_lst[i+1].num_neu):
                        sum_grad += self.layer_lst[i+1].layer[p].eg * self.layer_lst[i+1].layer[p].wt_lst[j]
                    self.layer_lst[i].layer[j].eg += sum_grad
                for k in range(self.layer_lst[i].layer[j].Num_ip):
                    if i == self.num_hidden:
                        err = tar[j] - out[j]
                        self.layer_lst[i].layer[j].wt_lst[k] += a * self.layer_lst[i].layer[j].ip_lst[k] * err
                    else:
                        self.layer_lst[i].layer[j].wt_lst[k] += a * self.layer_lst[i].layer[j].ip_lst[k] * self.layer_lst[i].layer[j].eg
                self.layer_lst[i].layer[j].bias += a * self.layer_lst[i].layer[j].eg
 
                        

        
    def sigmoid(val):
        x = math.exp(val)
        return x/(1+x)

    def func(value):
        return sigmoid(value)




nn = neural_net(2,1,1,2,0.8)
result = []
for i in range(1000):
    s = 0
    result = nn.train([1,1],0)
    s += math.pow(result[0]-0,2)
    result = nn.train([1,0],1)
    s += math.pow(result[0]-0,2)
    result = nn.train([0,1],1)
    s += math.pow(result[0]-0,2)
    result = nn.train([0,0],0)
    s += math.pow(result[0]-0,2)

print(s)
print(go(1,1,0,0)[0])


