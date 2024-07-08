import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lectura de datos
train_data = open('train.csv','r')
D = {}

for i, line in enumerate(train_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    D[i] = [float_values[0],sum(float_values,1)]

df = pd.DataFrame(D,index=[0,1])

dft=df.transpose()
print(dft.describe())


plt.scatter(dft[0],dft[1])
plt.show()