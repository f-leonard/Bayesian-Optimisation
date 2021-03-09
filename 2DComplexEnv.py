import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = pd.read_csv('Inputs_outputs.csv')
X = datasets.iloc[:, 0:2].values
Y = datasets.iloc[:, 3].values

weldarray = np.linspace(1000,4000,10)
clamparray = np.linspace(40,65,10)
XX,YY = np.meshgrid(weldarray,clamparray)
LSS = Y[0:100]
LSS = LSS.reshape(10,10)

rowcoordinates = np.dstack((XX, YY, LSS))

def function(x,y):
    for i in range(len(weldarray)):
        if x <= weldarray[i] and x>= weldarray[i-1]:
            x = weldarray[i-1]


        if y <= clamparray[i] and y>= clamparray[i-1]:
            y=clamparray[i-1]

    return x,y

print('The experiment coordinates are', function(1500,49))






#print(rowcoordinates[0][0][1])
#print(rowcoordinates[1][0][1])



coordinates = function(1500,55)
#print(coordinates)
def predictlSS():
    for j in range(len(weldarray)):
        if coordinates[1]== rowcoordinates[j][0][1]:
            print('its in the',rowcoordinates[j][0][1], 'column')












plt.figure(1)
plt.scatter(XX,YY)

plt.xlabel('Welding Energy(J)')
plt.ylabel('Vibration Amplitude ('r'$\mu$m)')
plt.title('2D Complex Environment Experiments')
plt.figure(2)
plt.contourf(XX,YY,LSS)
plt.colorbar()
plt.title('2D Complex Environment')
plt.xlabel('Welding Energy(J)')
plt.ylabel('Vibration Amplitude ('r'$\mu$m)')

plt.show()


