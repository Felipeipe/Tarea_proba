import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parte 4 Lectura de datos
train_data = open('train.csv','r')
D = {}

for i, line in enumerate(train_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    D[i] = [float_values[0],sum(float_values,1)]

df = pd.DataFrame(D,index=[0,1])

dft=df.transpose()

# fin

def covariance(X:np.array,Y:np.array) -> float:
    """ Entrega la covarianza de 2 variables aleatorias discretas

    Args:
        X (np.array): arreglo de valores posibles 1
        Y (np.array): arreglo de valores posibles 2

    Returns:
        float: entrega la covarianza
    """
    n = len(X)
    return -(np.mean(X) * np.mean(Y)) + (np.dot(X,Y) / n) 


# def lerp(X: np.array,Y: np.array):
#     """Genera una funcion que es una estimación lineal de X en función de Y,
#     es decir, devuelve una función de la forma X=AY+b

#     Args:
#         X (np.array): _description_
#         Y (np.array): _description_
#     """
print(dft.describe())
print(covariance(dft[0],dft[1]))
# plt.scatter(dft[0],dft[1])
# plt.show()