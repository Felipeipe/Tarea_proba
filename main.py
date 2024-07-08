import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parte 4 Lectura de datos
string_train_data = open('train.txt','r')
D = {}

for i, line in enumerate(string_train_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    D[i] = [float_values[0],sum(float_values,1)]

df = pd.DataFrame(D,index=[0,1])

train_data = df.transpose()

# fin

string_val_data = open('val.txt','r')
E = {}

for i, line in enumerate(string_val_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    E[i] = [float_values[0],sum(float_values,1)]

df2 = pd.DataFrame(D,index=[0,1])

val_data = df.transpose()

# Histogramas de datos de entrenamiento
plt.title('Histograma de porcentaje de carga de datos de entrenamiento')
plt.hist(train_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
plt.legend(loc='best')
plt.show()

plt.title('Histograma de Energía consumida en datos de entrenamiento')         
plt.hist(train_data[1],bins=500,alpha=0.5,label='Energía consumida',edgecolor = "b")     
plt.legend(loc='best')
plt.show()

# Histogramas de datos de validacion
plt.title('Histograma de porcentaje de carga de datos de validación')
plt.hist(val_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
plt.legend(loc='best')
plt.show()

plt.title('Histograma de Energía consumida en datos de validación')         
plt.hist(val_data[1],bins=500,alpha=0.5,label='Energía consumida',edgecolor = "b")     
plt.legend(loc='best')
plt.show()


