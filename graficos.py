from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

print(f"Covarianza de X e Y del entrenamiento: {covariance(train_data[0],train_data[1])}")
print(f"Coeficiente A del entrenamiento: {coef_A(train_data)}")
print(f"Coeficiente b del entrenamiento: {coef_b(train_data)}")
print(f'Error cuadrático medio de estimador lineal: {RMSE(LMMSE(train_data, val_data),val_data[0])}')
print(f'Error cuadrático medio de estimador no lineal: {RMSE(nonlinearEst(train_data[0],train_data[1],val_data[0],val_data[1],(10000,10000)),val_data[0])}')
print(f'Error absoluto medio de estimador lineal: {MAE(LMMSE(train_data, val_data),val_data[0])}')
print(f'Error absoluto medio de estimador no lineal: {MAE(nonlinearEst(train_data[0],train_data[1],val_data[0],val_data[1],(10000,10000)),val_data[0])}')

# # Crear una figura y ejes
# fig, axs = plt.subplots(3, 1)

# # Gráfico de la distribución de los datos de validación
# axs[0].set_title('Histograma de porcentaje de carga de datos de validación')
# axs[0].hist(val_data[0], bins=1000, alpha=0.5, label='Porcentaje de carga', color='tomato', edgecolor="tomato")
# axs[0].set_xlabel('Porcentaje de carga')
# axs[0].set_ylabel('Frecuencia por intervalo')
# axs[0].legend(loc='upper right')

# # Gráfico de la distribución de los datos estimados con el LMMSE
# axs[1].set_title('Histograma de porcentaje de carga de datos de validación con LMMSE (datos de entrenamiento)')
# axs[1].hist(LMMSE(train_data, val_data), bins=1000, alpha=0.5, label='Porcentaje de carga', color='tomato', edgecolor="tomato")
# axs[1].set_xlabel('Porcentaje de carga')
# axs[1].set_ylabel('Frecuencia por intervalo')
# axs[1].legend(loc='upper right')

# # Gráfico de la distribución de los datos estimados con el estimador no lineal
# axs[2].set_title('Histograma de porcentaje de carga de datos estimados con la esperanza de validación')
# axs[2].hist(nonlinearEst(train_data[0], train_data[1], val_data[0], val_data[1], (10000, 10000)), bins=1000, alpha=0.5, label='Porcentaje de carga', color='tomato', edgecolor="tomato")
# axs[2].set_xlabel('Porcentaje de carga')
# axs[2].set_ylabel('Frecuencia por intervalo')
# axs[2].legend(loc='upper right')

# # Ajustar el diseño para que no se solapen los gráficos
# plt.tight_layout(pad=3.0)

# # Mostrar la figura completa
# plt.show()
# # Crear una figura y ejes
# fig, axs = plt.subplots(2, 1)

# # Gráfico de error de estimador lineal de porcentaje de carga
# axs[0].set_title('Error de LMMSE ')
# axs[0].hist((LMMSE(train_data, val_data)-val_data[0]), bins=1000, alpha=0.5, label='Porcentaje de carga', color='tomato', edgecolor="tomato")
# axs[0].set_xlabel('Porcentaje de carga')
# axs[0].set_ylabel('Frecuencia por intervalo')
# axs[0].legend(loc='upper right')

# # Gráfico de la distribución de los datos de validación
# axs[1].set_title('Error de estimador no lineal')
# axs[1].hist((nonlinearEst(train_data[0], train_data[1], val_data[0], val_data[1], (10000, 10000))-val_data[0]), bins=1000, alpha=0.5, label='Porcentaje de carga', color='tomato', edgecolor="tomato")
# axs[1].set_xlabel('Error de porcentaje de carga')
# axs[1].set_ylabel('Frecuencia por intervalo')
# axs[1].legend(loc='upper right')

"""
11.-
Se puede notar que al observar el error como la diferencia
entre el estimador y val_data[0] (porcentaje de carga de datos
de validación) hay una cierta tendencia gaussiana, es decir,
el ruido presente dentro de los datos tiene una cierta distribución normal.

Este ruido puede ser visto en la distribución conjunta, ya que tiende
a agruparse alrededor de la recta de la identidad.
"""

# Ajustar el diseño para que no se solapen los gráficos
plt.tight_layout(pad=2.0)

# Mostrar la figura completa
plt.show()

