import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

# Funcion para la lectura de los datos
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
    return cov / var
def coef_b(dft):
    A = coef_A(dft)
    return dft[0].mean() - (A * dft[1].mean())

# Formula para la estimacion con el estimador lineal óptimo
def LMMSE(dftrain, dfval):
    return coef_A(dftrain)*dfval[1] + coef_b(dftrain)

# Calculamos la probabilidad de masa conjunta
def PMCF(X:np.array, Y: np.array, bins: tuple[int]) -> np.ndarray:
    """Entrega la probabilidad de masa conjunta al recibir
    las observaciones de 2 variables aleatorias

    Args:
        X (np.array): observación de variable aleatoria de largo n
        Y (np.array): segunda observación de variable aleatoria del mismo largo que X

    Returns:
        np.ndarray: probabilidad de masa conjunta
    """
    return np.histogram2d(X,Y,bins,density=True)

# Creamos la funcion para el estimador no lineal
def nonlinearEst(X, Y, Z, W, bin):
    h,x,y=PMCF(X, Y, bin)
    k,z,w=PMCF(Z, W, bin)
    x=x[:-1]
    y=y[:-1]
    z=z[:-1]
    w=w[:-1]
    numerador=np.dot(h,z)
    denominador=np.dot(h, np.ones(bin[0]))
    return numerador/denominador

# Formula para el Root Mean Square Error
def RMSE(X, Y):
    return np.sqrt((np.sum((X-Y)**2))/len(X))

# Formula para el Mean Absolute Error
def MAE(X, Y):
    return abs(np.sum((X-Y)))/len(X)

# Leemos las archivos .txt y los guardamos como data frame
train_data = lectura_de_datos('train.txt')
val_data = lectura_de_datos('val.txt')


###########
###########

X_train = train_data.iloc[:, 0].values  # Estado de carga final
Y_train_sum = train_data.iloc[:, 1].values  # Energía total consumida en cada ruta

# Definir los límites de los histogramas
x_min, x_max = X_train.min(), X_train.max()
y_min, y_max = Y_train_sum.min(), Y_train_sum.max()

# Crear un grid para los histogramas
num_bins = 50
x_bins = np.linspace(x_min, x_max, num_bins)
y_bins = np.linspace(y_min, y_max, num_bins)

# Calcular el histograma conjunto
H, xedges, yedges = np.histogram2d(X_train, Y_train_sum, bins=(x_bins, y_bins), density=True)

# Calcular el histograma marginal de Y
HY, yedges = np.histogram(Y_train_sum, bins=y_bins, density=True)

# Obtener los centros de los bins
x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
y_bin_centers = (yedges[:-1] + yedges[1:]) / 2

# Calcular el estimador no lineal E[X | Y=y]
x_hat = np.zeros_like(y_bin_centers)
for i, y in enumerate(y_bin_centers):
    joint_hist = H[:, i]
    marginal_y = HY[i]
    if marginal_y > 0:
        x_hat[i] = np.sum(joint_hist * x_bin_centers) / marginal_y
    else:
        x_hat[i] = 0

# Graficar el estimador no lineal
plt.figure(figsize=(10, 6))
plt.plot(y_bin_centers, x_hat/100, label='Estimador no lineal $E[X | Y=y]$', color='blue')
plt.xlabel('Y (Energía total consumida)')
plt.ylabel('X (Estado de carga final)')
plt.title('Estimador no lineal $E[X | Y=y]$')
plt.legend()
plt.show()