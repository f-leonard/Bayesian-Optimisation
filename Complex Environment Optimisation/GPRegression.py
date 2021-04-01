from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern,WhiteKernel
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)

'''Run this code to generate a plot of the predicted environment from the experiments generated in main.py'''

datacsv = pd.read_csv('Inputs_outputs.csv')
data = datacsv.iloc[:, 0].values
data = data[0:1000]
data = data[1::8]
data = data.reshape(5,5,5)

x = np.linspace(10, 40, 5)
y = np.linspace(40, 65, 5)
z = np.linspace(2, 4, 5)

'''This needs to be my interpolation function'''
interpolating_function = RegularGridInterpolator((x,y,z),data)
def f(x,y,z):
    """The function for conducitng experimentation."""
    pts = np.array([[x,y,z]])
    return float(interpolating_function(pts))
'''Read the experiments from the csv file'''

optpoints = pd.read_csv('Optpoints.csv',header = None)
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
#global error use 0.7
#max error use 1.3
kernel = Matern(nu=1.3)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25,alpha=3e-4)

gp.fit(X, y)
testdata = []
functiondata = []
squarederror = []

def predictiontarget(x,y,z):
    return float(gp.predict([[x,y,z]]))

'''Execute line 60 for the global RMSE and line 61 for optimum region rmse'''
#testarray = np.array(pd.read_csv('Experiments for MSE.csv'))
testarray = np.array(pd.read_csv('Experiments for optimum region MSE.csv'))

for q in range(len(testarray)):
    testdata.append(predictiontarget(testarray[q][0] / 100, testarray[q][1], testarray[q][2]))
    functiondata.append(f(testarray[q][0] / 100, testarray[q][1], testarray[q][2]))

for r in range(len(testarray)):
    squarederror.append(((functiondata[r] - testdata[r]) ** 2))
meansquarederror = sum(squarederror) / len(testarray)
print('The root mean squared error is ', np.sqrt(meansquarederror))




a = []
b = []
j = []
cv = []

def slicex(x):
    for i in range(50000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        c = predictiontarget(x, y, z)

        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)

def slicey(y):
    for i in range(50000):
        x = np.random.uniform(10,40)
        y = y
        z = np.random.uniform(2, 4)
        c = predictiontarget(x, y, z)

        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)


def slicez(z):
    for i in range(50000):
        x = np.random.uniform(10,40)
        y = np.random.uniform(40, 65)
        z = z
        c = predictiontarget(x, y, z)

        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
slicex(24.62)
slicey(52.44)
slicez(3.599)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
ax.view_init(elev=13., azim=-140)


ax.set_xlabel('Welding Energy (J)',fontsize = 14)
ax.set_ylabel('Vibration amplitude ('r'$\mu$m)',fontsize = 14)
ax.set_zlabel('Clamping pressure (bar)', fontsize = 14)
plt.title('Predicted Environment - 5 Random, 4 EI Steps', fontsize = 18)
fig.figsize = (30,30)
cbar1 = fig.colorbar(img)
cbar1.set_label('LSS (N)', labelpad = -40, y = 1.05, rotation = 0)
plt.show()

