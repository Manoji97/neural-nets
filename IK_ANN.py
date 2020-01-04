import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd




def find_XY(deta,l1,l2):
    x_o = l1*np.cos(np.deg2rad(deta[0])) + l2*np.cos(np.deg2rad(deta[1]))
    y = l1*np.sin(np.deg2rad(deta[0])) + l2*np.sin(np.deg2rad(deta[1]))
    x = x_o * (np.cos(np.deg2rad(deta[2])))
    z = x_o * (np.sin(np.deg2rad(deta[2])))
    return [round(x,2),round(y,2),round(z,2)]

l1 = 1
l2 = 1
Training_data = []
for i in range(0,181):
    for j in range(0,181):
        for k in range(0,181):
            if k%30 == 0:
                a = find_XY([i,j,k],l1,l2)
                Training_data.append([[i,j,k],a])



with open("IK_inputs_new","wb") as file:
    pickle.dump(Training_data,file)
    
#visualization
l = []
for i in range(100000):
    l.append(Training_data[i][1])

df = pd.DataFrame(l,dtype = float,columns = ["x","y","z"])


    
df["x"][:10]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["x"],df["z"],df["y"])
    
plt.scatter(df["x"],df["y"],df["z"])  
    
    
    
    
    
    
    
    
    
    
    
    

def sig(x):
    return 1/(1+np.exp(-x))
sigmoid = np.vectorize(sig)


# Normalization
def red_180(a):
    y = [a[0]/180,a[1]/180]
    return y

def red_4(q):
    y = [q[0]/4,q[1]/4,q[2]/180]
    return y
x = []

for i in Y:
    x.append(red_4(i))

for i in Training_data:
    x.append(red_180(i[0]))

        




















