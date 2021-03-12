from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization

datacsv = pd.read_csv('Inputs_outputs.csv')
data = datacsv.iloc[:, 0].values
data = data[0:1000]
data = data[1::8]
data = data.reshape(5,5,5)

x = np.linspace(1000, 4000, 5)
y = np.linspace(40, 65, 5)
z = np.linspace(2, 4, 5)
#data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))


interpolating_function = RegularGridInterpolator((x,y,z),data)

def function(x,y,z):
    pts = np.array([[x, y, z]])

    return float(interpolating_function(pts))

a = []
b = []
j = []
cv = []

def slicex(x):
    for i in range(10000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        c = function(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1
def slicey(y):
    for i in range(10000):
        x = np.random.uniform(1000,4000)
        y = y
        z = np.random.uniform(2, 4)
        c = function(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1

def slicez(z):
    for i in range(10000):
        x = np.random.uniform(1000,4000)
        y = np.random.uniform(40, 65)
        z = z
        c = function(x, y, z)

        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)


'''for i in range(100000):
    x = np.random.uniform(2450, 2600)
    y = np.random.uniform(50, 54)
    z = np.random.uniform(3,3.7)
    c = function(x, y, z)

    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)'''

slicex(2500.1905)
slicey(52.50137)
slicez(3.50206)

'''Max value is 3212'''
max_element = np.where(cv == np.amax(cv))

max_index = int(max_element[0])
print('The global max is located at ',a[max_index],b[max_index],j[max_index])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
ax.view_init(elev=13., azim=-140)
fig.colorbar(img)
print('The maximum value observed on the plot was', max(cv))
#print('The maximum value observed by bayesian optimisation was', np.max(datalist))
ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')

plt.savefig('4D plot of environment')
plt.show()
