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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = []
b = []
j = []
cv = []

def slicex(x):
    for i in range(20000):
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
    for i in range(20000):
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
    for i in range(20000):
        x = np.random.uniform(1000,4000)
        y = np.random.uniform(40, 65)
        z = z
        c = function(x, y, z)
        i = i + 1
        a.append(x)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1


pbounds = {'y': (40, 65), 'z': (2, 4),'x':(1000,4000)}

optimizer = BayesianOptimization(
    f=function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=456165
)
def probe_point(x,y,z):
    '''This is a function that allows the user to probe specific points in the environment
    and add them to the environment observations'''
    return optimizer.probe(params={"x": x, "y": y, "z":z},lazy=True,)


optimizer.maximize(init_points =5, n_iter = 5,acq = 'ei',kappa=5,xi = 0.3,alpha = 1,normalize_y = True)

xlist = []
ylist = []
zlist = []
datalist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))
    zlist.append(float(res[1].get('params').get('z')))
    datalist.append(float(res[1].get('target')))

'''img = ax.scatter(xlist,ylist,zlist,marker='o',c='black')'''
datalist = np.array(datalist)
max_element = np.where(datalist == np.amax(datalist))

max_index = int(max_element[0])
slicex(xlist[max_index])
slicey(ylist[max_index])
slicez(zlist[max_index])



optpointsarray = np.array((list((zip(xlist,ylist,zlist)))))
np.savetxt('Optpoints.csv',optpointsarray,delimiter=',')

img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
ax.view_init(elev=13., azim=-140)
fig.colorbar(img)
print('The maximum value observed on the plot was', max(cv))
print('The maximum value observed by bayesian optimisation was', np.max(datalist))
ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')
plt.title('Environment Sliced Through Maximum Observation Point')
plt.savefig('4D plot of environment')
plt.show()

