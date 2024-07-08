import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = open('train.csv','r')
D = {}

for line in train_data:
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    D[float_values[0]] = sum(float_values,1)

df = pd.DataFrame(D,index=[0])
    
print(df)

print(sum([8140.13955009624,307.5538151535111,0.809257308834276,1430.7527648043197,0.0,17784.312082843033]))