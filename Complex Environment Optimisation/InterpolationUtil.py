from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization,UtilityFunction

'''This section reads in the fabricated experimental data from  a csv file, returns 125 experiments and creates a 
5 x 5 x 5 array from the experiments to interpolate the function over'''
datacsv = pd.read_csv('Inputs_outputs.csv')
data = datacsv.iloc[:, 0].values
data = data[0:1000]
data = data[1::8]
data = data.reshape(5,5,5)

x = np.linspace(10, 40, 5)
y = np.linspace(40, 65, 5)
z = np.linspace(2, 4, 5)
#data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))


interpolating_function = RegularGridInterpolator((x,y,z),data)




def function(x,y,z):
    pts = np.array([[x, y, z]])

    return float(interpolating_function(pts))

utility_function = UtilityFunction(kind="ei",xi=0.5,kappa=0)


a = []
b = []
j = []
cv = []

def utilitytarget(xtarget,ytarget,ztarget):
    xyparam = np.array([[xtarget,ytarget,ztarget]])
    return float((utility_function.utility(xyparam, optimizer._gp, 0)))



def slicex(x):
    for i in range(50000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        c = utilitytarget(x, y, z)
        i = i + 1
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1
def slicey(y):
    for i in range(50000):
        x = np.random.uniform(10,40)
        y = y
        z = np.random.uniform(2, 4)
        c = utilitytarget(x, y, z)
        i = i + 1
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1

def slicez(z):
    for i in range(50000):
        x = np.random.uniform(10,40)
        y = np.random.uniform(40, 65)
        z = z
        c = utilitytarget(x, y, z)
        i = i + 1
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        i = i + 1
def intersect(x,y,z):
    return slicex(x), slicey(y),slicez(z)

pbounds = {'y': (40, 65), 'z': (2, 4),'x':(10,40)}
optimizer = BayesianOptimization(
    f=function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    #random_state=456165
    random_state=2
)
def probe_point(x,y,z):
    return optimizer.probe(params={"x": x, "y": y, "z":z},lazy=True,)

'''probe_point(4000,65,4)
probe_point(4000,65,2)
probe_point(3000,40,2)
probe_point(1000,40,2)
probe_point(2500,52.5,3)'''

optimizer.maximize(init_points =5, n_iter = 5,acq = 'ei',kappa=5,xi = 0.4,alpha = 3e-4,normalize_y = True)
optimizer.maximize(init_points =0, n_iter = 5,acq = 'poi',kappa=5,xi = 0,alpha = 3e-4,normalize_y = True)

xlist = []
ylist = []
zlist = []
datalist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))
    zlist.append(float(res[1].get('params').get('z')))
    datalist.append(float(res[1].get('target')))

'''ax.scatter(xlist,ylist,zlist,marker='o')'''
datalist = np.array(datalist)
max_element = np.where(datalist == np.amax(datalist))


optpointsarray = np.array((list((zip(xlist,ylist,zlist)))))
np.savetxt('Optpoints.csv',optpointsarray,delimiter=',')

max_index = int(max_element[0])
slicex(xlist[max_index])
slicey(ylist[max_index])
slicez(zlist[max_index])

'''Select Planes for slicing'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
ax.view_init(elev=13., azim=-140)
fig.colorbar(img)
print('The maximum value observed on the plot was', max(cv))
print('The maximum value observed by bayesian optimisation was', np.max(datalist))
ax.set_xlabel('Welding Energy (J)',fontsize = 14)
ax.set_ylabel('Vibration amplitude(um)', fontsize = 14)
ax.set_zlabel('Clamping pressure (bar)', fontsize = 14)
plt.title('Final Utility Function',fontsize = 18)
plt.show()
