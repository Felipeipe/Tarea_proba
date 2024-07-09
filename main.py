import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random

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
    return cov * var

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


# gráficos analisis preliminar

plt.rcParams['figure.figsize'] = [10, 6]

# # Histogramas de datos de entrenamiento
# plt.title('Histograma de porcentaje de carga de datos de entrenamiento')
# plt.hist(train_data[0],bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.grid(True)
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()

# plt.title('Histograma de Energía consumida en datos de entrenamiento')         
# plt.hist(train_data[1],bins=500,alpha=0.5,label='Energía consumida [Wh]',edgecolor = "b")
# plt.xlabel('Energía consumida [Wh]')
# plt.ylabel('Frecuencia por intervalo')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# # Histogramas de datos de validacion
# plt.title('Histograma de porcentaje de carga de datos de validación')
# plt.hist(val_data[0],bins=50,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.ylabel('Frecuencia por intervalo')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# plt.title('Histograma de Energía consumida en datos de validación')         
# plt.hist(val_data[1],bins=100,alpha=0.5,label='Energía consumida',edgecolor = "b")   
# plt.xlabel('Energía consumida [Wh]')
# plt.ylabel('Frecuencia por intervalo')  
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# # plt.scatter(train_data[0], train_data[0], s=0.01, color='blue', label='Datos de entrenamiento')
# # #plt.scatter(LMMSE(train_data), train_data[0], s=0.001, color='red', label='Estimador lineal')
# # plt.title(r'Distribución conjunta $f_{}$')
# # plt.legend(loc='best')
# # plt.grid(True)
# # plt.show()

# # plt.title(r'Distribución conjunta $f_{X,Y}(x,y)$')         
# # plt.scatter(train_data[0],train_data[1],label="Distribución conjunta", s=0.1)
# # plt.xlabel("porcentaje de carga")
# # plt.ylabel("energía consumida [Wh]")
# # plt.grid(True)
# # plt.legend(loc='best')
# # plt.show()

# # Grafico de estimador lineal
# # plt.title(r'Distribución conjunta $f_{\hat{X},X}$')         
# # plt.scatter(LMMSE(train_data),train_data[0],label='distribución conjunta',s=0.1)
# # plt.xlabel('porcentaje de carga')
# # plt.ylabel('porcentaje de carga')
# # plt.grid(True)
# # plt.legend(loc='best')
# # plt.show()

# # Histogramas de datos de entrenamiento con LMMSE
# plt.title('Histograma de porcentaje de carga de datos de entrenamiento con LMMSE')
# plt.hist(LMMSE(train_data),bins=100,alpha=0.5,label='Porcentaje de carga',edgecolor = "b")
# plt.xlabel('Porcentaje de carga')
# plt.grid(True)
# plt.ylabel('Frecuencia por intervalo')
# plt.legend(loc='best')
# plt.show()





XY = np.stack((train_data[0],train_data[1]),axis=-1)

# def selection(XY, limitXY=[[0,+1],[0,+100000]]):
#         XY_select = []
#         for elt in XY:
#             if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
#                 XY_select.append(elt)

#         return np.array(XY_select)

# XY_select = selection(XY, limitXY=[[0,+1],[0,+100000]])


# xAmplitudes = np.array(XY_select)[:,0]#your data here
# yAmplitudes = np.array(XY_select)[:,1]#your other data here


# fig = plt.figure() #create a canvas, tell matplotlib it's 3d
# ax = fig.add_subplot(111, projection='3d')


# hist, xedges, yedges = np.histogram2d(train_data[0], train_data[1], bins=(100,100), range = [[0,+1],[0,+100000]]) # you can change your bins, and the range on which to take data
# # hist is a 7X7 matrix, with the populations for each of the subspace parts.
# xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) -(xedges[1]-xedges[0])


# xpos = xpos.flatten()*1./2
# ypos = ypos.flatten()*1./2
# zpos = np.zeros_like (xpos)

# dx = xedges [1] - xedges [0]
# dy = yedges [1] - yedges [0]
# dz = hist.flatten()

# max_height = 80   # get range of colorbars so we can normalize
# min_height = 0
# # scale each z to [0,1], and get their rgb values


# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue', zsort='average')
# plt.title("Probabilidad de grasa :v conjunta de las variables X y Y")
# plt.xlabel("X - Porcentaje de carga restante")
# plt.ylabel("Y - Energía consumida en el viaje (Wh)")
# plt.savefig("Frecuencia conjunta de las variables X y Y")
# plt.show()

print(XY)

Z=np.linspace(0, 1, 100)
W=np.linspace(0,100000, 500)