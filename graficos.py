from main import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

LMMSE_data = LMMSE(train_data, val_data)
nonlinearEst_data = nonlinearEst(train_data[0], train_data[1], val_data[0], val_data[1], (10000, 10000))

print(f"Covarianza de X e Y del entrenamiento: {covariance(train_data[0],train_data[1])}")
print(f"Coeficiente A del entrenamiento: {coef_A(train_data)}")
print(f"Coeficiente b del entrenamiento: {coef_b(train_data)}")
print(f'Error cuadrático medio de estimador lineal: {RMSE(LMMSE_data,val_data[0])}')
print(f'Error cuadrático medio de estimador no lineal: {RMSE(nonlinearEst_data,val_data[0])}')
print(f'Error absoluto medio de estimador lineal: {MAE(LMMSE_data,val_data[0])}')
print(f'Error absoluto medio de estimador no lineal: {MAE(nonlinearEst_data,val_data[0])}')

fig, axs = plt.subplots(3, 1)

# Gráfico de dispersión de los datos de validación
axs[0].set_title('Gráfico de dispersión del porcentaje de carga de datos de validación')
axs[0].scatter(val_data[1], val_data[0], alpha=0.5, color='lightseagreen', edgecolor="lightseagreen", s=0.1)
axs[0].set_xlabel('Energía consumida')
axs[0].set_ylabel('Porcentaje de carga')

# Gráfico de dispersión de los datos estimados con el LMMSE
axs[1].set_title('Gráfico de dispersión del porcentaje de carga de datos de validación estimados con LMMSE')
axs[1].scatter(val_data[1], LMMSE_data, alpha=0.5, color='darkolivegreen', edgecolor="darkolivegreen", s=0.1)
axs[1].set_xlabel('Energía consumida')
axs[1].set_ylabel('Porcentaje de carga')

# Gráfico de dispersión de los datos estimados con el estimador no lineal
axs[2].set_title('Gráfico de dispersión del porcentaje de carga de datos de validación estimados con la Esperanza Condicional')
axs[2].scatter(val_data[1], nonlinearEst_data, alpha=0.5, color='tomato', edgecolor="tomato", s=0.1)
axs[2].set_xlabel('Energía consumida')
axs[2].set_ylabel('Porcentaje de carga')

plt.tight_layout(pad=3.0)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1)
# Gráfico de error de estimador lineal de porcentaje de carga
axs[0].set_title('Gráfico de dispersión del error de porcentaje de carga de datos de validación estimados con LMMSE')
axs[0].scatter(val_data[1], LMMSE_data-val_data[0], alpha=0.5, color='darkolivegreen', edgecolor="darkolivegreen", s=0.1)
axs[0].set_xlabel('Energía consumida')
axs[0].set_ylabel('Error porcentaje de carga')

# Gráfico de la distribución de los datos de validación
axs[1].set_title('Gráfico de dispersión del error de porcentaje de carga de datos estimados con la esperanza de validación')
axs[1].scatter(val_data[1], nonlinearEst_data-val_data[0], alpha=0.5, color='tomato', edgecolor="tomato", s=0.1)
axs[1].set_xlabel('Energía consumida')
axs[1].set_ylabel('Error porcentaje de carga')

plt.tight_layout(pad=2.0)
plt.tight_layout()
plt.show()

# Histograma de error de estimador lineal de porcentaje de carga
plt.title('Error de LMMSE ')
plt.hist((nonlinearEst_data-val_data[0]), bins=1000, alpha=0.5, label='LMMSE', color='tomato', edgecolor="tomato")
plt.hist((LMMSE_data-val_data[0]), bins=1000, alpha=0.5, label='Esperanza Condicional', color='darkolivegreen', edgecolor="darkolivegreen")
plt.xlabel('Porcentaje de carga')
plt.ylabel('Frecuencia por intervalo')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

"""
11.-
Se puede notar que al observar el error como la diferencia
entre el estimador y val_data[0] (porcentaje de carga de datos
de validación) hay una cierta tendencia gaussiana, es decir,
el ruido presente dentro de los datos tiene una cierta distribución normal.

Este ruido puede ser visto en la distribución conjunta, ya que tiende
a agruparse alrededor de la recta de la identidad.
"""