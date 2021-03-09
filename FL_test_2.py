import numpy as np


import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.interpolate import RegularGridInterpolator
import pandas as pd


datacsv = pd.read_csv('Inputs_outputs.csv')
data = datacsv.iloc[:, 3].values
data = data[5000:6000]
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

'''This is where I put in the optimisation points'''
optpoints = pd.read_csv('Optpoints.csv')
optpoints = np.array(optpoints)

X = optpoints

yarray =[]
for i in range(len(X)):
    yarray.append(f(X[i][0],X[i][1],X[i][2]))
yarray = np.array(yarray).reshape(-1,1)



#X = []

y = yarray

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
'''This needs to be the 3 parameters'''
weldenergy = np.linspace(1000,4000,100)
vibamp = np.linspace(40,65,100)
clamppress = np.linspace(2,4,100)
pickme = np.meshgrid(weldenergy,vibamp,clamppress)



# Instantiate a Gaussian Process model
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=150)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

'''This is where I put in the continuous values to populate the plot'''
X_pred = []
a = []
b = []
c = []
cv = []
'''for j in range(100000):
    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    X_pred.append([x,y,z])
    a.append(x)
    b.append(y)
    c.append(z)'''
def slicex(x):
    for i in range(50000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        X_pred.append([x,y,z])

        a.append(x)
        b.append(y)
        c.append(z)
        cv.append(c)

def slicey(y):
    for i in range(50000):
        x = np.random.uniform(1000,4000)
        y = y
        z = np.random.uniform(2, 4)
        X_pred.append([x,y,z])

        a.append(x)
        b.append(y)
        c.append(z)
        cv.append(c)

def slicez(z):
    for i in range(50000):
        x = np.random.uniform(1000, 4000)
        y = np.random.uniform(40, 65)
        z = z
        X_pred.append([x, y, z])

        a.append(x)
        b.append(y)
        c.append(z)
        cv.append(c)

slicex(3191)
'''slicey(59.7476)
slicez(2.3779)'''



# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(pickme, return_std=True)

y_pred= np.array(y_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(a, b, c,c=y_pred, cmap=plt.jet())
fig.colorbar(img)

ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')

plt.savefig('4D plot of environment')
plt.show()
