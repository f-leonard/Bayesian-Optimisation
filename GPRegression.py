from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern,WhiteKernel
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)
datacsv = pd.read_csv('Inputs_outputs.csv',header = None)
data = datacsv.iloc[:, 0].values
data = data[0:1000]
data = data[1::8]
data = data.reshape(5,5,5)

x = np.linspace(1000, 4000, 5)
y = np.linspace(40, 65, 5)
z = np.linspace(2, 4, 5)
'''This needs to be my interpolation function'''
interpolating_function = RegularGridInterpolator((x,y,z),data)
def f(x,y,z):
    """The function to predict."""
    pts = np.array([[x,y,z]])
    return float(interpolating_function(pts))
optpoints = pd.read_csv('Optpoints.csv')
optpoints = np.array(optpoints)

X = optpoints


# Input space
x1 = np.linspace(X[:,0].min(), X[:,0].max())
x2 = np.linspace(X[:,1].min(), X[:,1].max())
x3 = np.linspace(X[:,2].min(), X[:,2].max())
x = (np.array([x1, x2, x3])).T

y = []
for i in range(len(X)):
    y.append(f(X[i][0],X[i][1],X[i][2]))

y = np.array(y)

kernel = C(1.0, (1, 1e3)) * Matern([1,1,1], (1e-5, 1e5))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30,alpha=1)

gp.fit(X, y)


testdata = []
functiondata = []
squarederror = []

def predictiontarget(x,y,z):
    return float(gp.predict([[x,y,z]]))

testarray = np.array(pd.read_csv('Experiments for MSE.csv'))

for q in range(len(testarray)):
    testdata.append(predictiontarget(testarray[q][0],testarray[q][1],testarray[q][2]))
    functiondata.append(f(testarray[q][0],testarray[q][1],testarray[q][2]))


for r in range(len(testarray)):
    squarederror.append(((functiondata[r]-testdata[r])**2))
meansquarederror = sum(squarederror)/len(testarray)


print('The mean squared error is ',meansquarederror)



a = []
b = []
j = []
cv = []





def slicex(x):
    for i in range(20000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        c = predictiontarget(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1
def slicey(y):
    for i in range(20000):
        x = np.random.uniform(1000,4000)
        y = y
        z = np.random.uniform(2, 4)
        c = predictiontarget(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1

def slicez(z):
    for i in range(20000):
        x = np.random.uniform(1000,4000)
        y = np.random.uniform(40, 65)
        z = z
        c = predictiontarget(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1
slicex(2392)
slicey(52.2035)
slicez(3.5758)

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
plt.title('Bayesian Optimisation Predicted Environment')
plt.show()