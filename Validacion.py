from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

print(train_data.describe())
print(val_data.describe())
print(f"Covarianza de X e Y del entrenamiento: {covariance(train_data[0],train_data[1])}")
print(f"Coeficiente A del entrenamiento: {coef_A(train_data)}")
print(f"Coeficiente b del entrenamiento: {coef_b(train_data)}")

#Grafico de la distribucion de los datos de validacion
plt.title('Histograma de porcentaje de carga de datos de validación')
plt.hist(val_data[0],bins=50,alpha=0.5,label='Porcentaje de carga',color='tomato',edgecolor = "tomato")
plt.xlabel('Porcentaje de carga')
plt.ylabel('Frecuencia por intervalo')
plt.legend(loc='best')
plt.show()

#Grafico de la distribucion de los datos estimados con el LMMSE
plt.title('Histograma de porcentaje de carga de datos de validación con LMMSE (datos de entrenamiento)')
plt.hist(LMMSE(train_data, val_data),bins=50,alpha=0.5,label='Porcentaje de carga',color='tomato',edgecolor = "tomato")
plt.xlabel('Porcentaje de carga')
plt.ylabel('Frecuencia por intervalo')
plt.legend(loc='best')
plt.show()

#Grafico de la distribucion de los datos estimados con el estimador no lineal
