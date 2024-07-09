import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parte 4 Lectura de datos
string_train_data = open('train.txt', 'r')
D = {}

for i, line in enumerate(string_train_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    D[i] = [float_values[0],sum(float_values,1)]
df = pd.DataFrame(D,index=[0,1])
train_data = df.transpose()

string_val_data = open('val.txt','r')
E = {}

for i, line in enumerate(string_val_data):
    str_values = line.split(',')
    float_values = [float(x) for x in str_values]
    
    E[i] = [float_values[0],sum(float_values,1)]
df2 = pd.DataFrame(E,index=[0,1])
val_data = df2.transpose()

# Funcion de covarianza
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

# Calculamos los coeficientes del estimador lineal óptimo
def coef_A(dft):
    cov = covariance(dft[0],dft[1])
    var = dft[1].var()
    return cov / var

def coef_b(dft):
    A = coef_A(dft)
    return dft[0].mean() - (A * dft[1].mean())

# Creamos el estimador lineal óptimo
def LMMSE(dft):
    return coef_A(dft)*dft[1] + coef_b(dft)

# Mostramos todos los Medidas de tendencia central
print(train_data.describe())
print(val_data.describe())
print(f"Covarianza de X e Y del entrenamiento: {covariance(train_data[0],train_data[1])}")
print(f"Coeficiente A del entrenamiento: {coef_A(train_data)}")
print(f"Coeficiente b del entrenamiento: {coef_b(train_data)}")
print(f"Estimador lineal x(Y) del entrenamiento: \n{LMMSE(train_data)}")
#print(LMMSE(train_data)-train_data[0])

# Histogramas de datos de entrenamiento
plt.title('Histograma de porcentaje de carga de datos de entrenamiento')
plt.hist(train_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
plt.legend(loc='best')
plt.show()

plt.title('Histograma de Energía consumida en datos de entrenamiento')         
plt.hist(train_data[1],bins=500,alpha=0.5,label='Energía consumida',edgecolor = "b")     
plt.legend(loc='best')
plt.show()

plt.scatter(train_data[0], train_data[0], s=0.001, color='blue', label='Datos de entrenamiento')
#plt.scatter(LMMSE(train_data), train_data[0], s=0.001, color='red', label='Estimador lineal')
plt.title('Histograma de porcentaje de carga de datos de entrenamiento con estimador lineal')
plt.legend(loc='best')
plt.show()

plt.title('Histograma de porcentaje de carga de datos de entrenamiento')         
plt.scatter(train_data[0],train_data[1], s=0.001)     
plt.legend(loc='best')
plt.show()

plt.title('Histograma de porcentaje de carga de datos de entrenamiento con estimador lineal')         
plt.scatter(LMMSE(train_data),train_data[0],s=0.001)     
plt.legend(loc='best')
plt.show()

# Histogramas de datos de entrenamiento con LMMSE
plt.title('Histograma de porcentaje de carga de datos de entrenamiento con LMMSE')
plt.hist(LMMSE(train_data),bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
plt.legend(loc='best')
plt.show()

# Histogramas de datos de validacion
#plt.title('Histograma de porcentaje de carga de datos de validación')
#plt.hist(val_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
#plt.legend(loc='best')
#plt.show()
#
#plt.title('Histograma de Energía consumida en datos de validación')         
#plt.hist(val_data[1],bins=500,alpha=0.5,label='Energía consumida',edgecolor = "b")     
#plt.legend(loc='best')
#plt.show()


