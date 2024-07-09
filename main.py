import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

# parte 4 Lectura de datos
def lectura_de_datos(path:str)->pd.DataFrame:
    """Lee un archivo de formato .txt y lo convierte a un DataFrame

    Args:
        path (str): nombre del archivo

    Returns:
        pd.DataFrame: dataframe de 2 columnas
    """
    
    string_data = open(path, 'r')
    D = {}

    for i, line in enumerate(string_data):
        str_values = line.split(',')
        float_values = [float(x) for x in str_values]
        
        D[i] = [float_values[0],sum(float_values,1)]
    df = pd.DataFrame(D,index=[0,1])
    return df.transpose()

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
    return cov * var

def coef_b(dft):
    A = coef_A(dft)
    return dft[0].mean() - (A * dft[1].mean())

# Creamos el estimador lineal óptimo
def LMMSE(dft):
    return coef_A(dft)*dft[1] + coef_b(dft)

def PMCF(X:np.array, Y: np.array, bins: tuple[int]) -> np.ndarray:
    """entrega la probabilidad de masa conjunta al recibir
    las observaciones de 2 variables aleatorias

    Args:
        X (np.array): observación de variable aleatoria de largo n
        Y (np.array): segunda observación de variable aleatoria del mismo largo que X

    Returns:
        np.ndarray: probabilidad de masa conjunta
    """
    return np.histogram2d(X,Y,bins,density=True)


def nonlinearEst(X, Y, n):
    h,x,y=PMCF(X, Y, n)
    x=x[:-1]
    y=y[:-1]
    print(len(x))
    print(len(y))
    numerador=np.dot(h,x)
    denominador=np.dot(h, np.ones(n[0]))
    return numerador/denominador

def recta(a:float,X: np.array) -> float:
    return np.ones(len(X)) * a

def err(X, Y):
    return abs(X - Y)

def RMSE(X, Y):
    return np.sqrt((np.sum((X-Y)**2))/len(X))

def MAE(X, Y):
    return np.sum((X-Y))/len(X)


train_data = lectura_de_datos('train.txt')
val_data = lectura_de_datos('val.txt')


# Mostramos todos los Medidas de tendencia central
print(train_data.describe())
print(val_data.describe())
print(f"Covarianza de X e Y del entrenamiento: {covariance(train_data[0],train_data[1])}")
print(f"Coeficiente A del entrenamiento: {coef_A(train_data)}")
print(f"Coeficiente b del entrenamiento: {coef_b(train_data)}")
# print(f"Estimador lineal x(Y) del entrenamiento: \n{LMMSE(train_data)}")

# gráficos analisis preliminar


# # Histogramas de datos de entrenamiento
# plt.title('Histograma de porcentaje de carga de datos de entrenamiento')
# plt.hist(train_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',color='tomato',edgecolor = "tomato")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# plt.title('Histograma de Energía consumida en datos de entrenamiento')         
# plt.hist(train_data[1],bins=500,alpha=0.5,label='Energía consumida [Wh]',color='g',edgecolor = "g")
# plt.xlabel('Energía consumida [Wh]')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# # Histogramas de datos de validacion
# plt.title('Histograma de porcentaje de carga de datos de validación')
# plt.hist(val_data[0],bins=50,alpha=0.5,label='Porcentaje de carga',color='tomato',edgecolor = "tomato")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# plt.title('Histograma de Energía consumida en datos de validación')         
# plt.hist(val_data[1],bins=100,alpha=0.5,label='Energía consumida',color='g',edgecolor = "g")   
# plt.xlabel('Energía consumida [Wh]')
# plt.ylabel('Frecuencia por intervalo')  
# plt.legend(loc='best')
# plt.show()

# # plt.scatter(train_data[0], train_data[0], s=0.01, color='blue', label='Datos de entrenamiento')
# plt.scatter(train_data[0],LMMSE(train_data), s=0.1, color='tomato', label='Estimador lineal')
# plt.title(r'Distribución conjunta $f_{}$')
# plt.legend(loc='best')
# plt.show()

# plt.title(r'Distribución conjunta $f_{X,Y}(x,y)$')         
# plt.scatter(train_data[0],train_data[1],color='mediumpurple',label="Distribución conjunta", s=0.1)
# plt.xlabel("porcentaje de carga")
# plt.ylabel("energía consumida [Wh]")
# plt.legend(loc='best')
# plt.show()

# X=np.linspace(0,1,1000)


# Grafico de estimador lineal
# plt.title(r'Distribución conjunta $f_{\hat{X},X}$')         
# plt.scatter(train_data[0],LMMSE(train_data),color='orchid',label='distribución conjunta',s=0.1)
# plt.xlabel('Porcentaje de carga (real)')
# plt.ylabel('Porcentaje de carga (LMMSE)')
# plt.legend(loc='best')
# plt.show()

# # Histogramas de datos de entrenamiento con LMMSE
# plt.title('Histograma de porcentaje de carga de datos de entrenamiento con LMMSE')
# plt.hist(LMMSE(train_data),bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()



# plt.title('Error entre datos de validación y estimador lineal')
# plt.scatter(val_data[0], err(LMMSE(val_data),val_data[0]),label='Porcentaje de carga',color = "b", s=0.1)
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# plt.title('Error entre datos de validación y estimador ideal')
# plt.hist(err(val_data[0], nonlinearEst(val_data[0], val_data[1], (10000, 10000))),bins=1000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# plt.title('RMSE entre datos de validación y estimador lineal')
# plt.hist(RMSE(val_data[0], LMMSE(val_data)),bins=1000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Magnitud del RMSE')
# plt.legend(loc='best')
# plt.show()

# plt.title('MAE entre datos de validación y estimador lineal')
# plt.hist(MAE(val_data[0], LMMSE(val_data)),bins=1000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Magnitud del MAE')
# plt.legend(loc='best')
# plt.show()

# plt.title('RMSE entre datos de validación y estimador ideal')
# plt.hist(RMSE(val_data[0], nonlinearEst(val_data[0], val_data[1], (10000, 10000))),bins=1000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Magnitud del RMSE')
# plt.legend(loc='best')
# plt.show()

# plt.title('MAE entre datos de validación y estimador ideal')
# plt.hist(MAE(val_data[0], nonlinearEst(val_data[0], val_data[1], (10000, 10000))),bins=1000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Magnitud del MAE')
# plt.legend(loc='best')
# plt.show()


plt.title('MAE entre datos de validación y estimador ideal')
plt.hist(nonlinearEst(val_data[0], val_data[1], (1000, 1000)),bins=10000,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
plt.xlabel('Porcentaje de carga')
plt.ylabel('Magnitud del MAE')
plt.legend(loc='best')
plt.show()