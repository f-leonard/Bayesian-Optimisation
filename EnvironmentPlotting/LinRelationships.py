import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets = pd.read_csv('Inputs_outputs.csv')
Y = datasets.iloc[:, 3].values
Y = list(Y)
Y.append(500)
Y = np.array(Y)

Y = Y.reshape(10,10)
weldingenergy = np.linspace(1000,4000,10)
vibrationamp = np.linspace(40,65,10)
clampingpres = np.linspace(2,4,10)

XX,YY= np.meshgrid(weldingenergy,vibrationamp)
data = XX**2-YY**4
print(data.shape)
plt.figure(1)
plt.scatter(XX,YY)
plt.figure(2)
plt.contourf(YY,XX,Y)
plt.colorbar()
plt.figure(3)
plt.plot(np.ravel(Y))
plt.show()

