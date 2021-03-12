from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ThreeDEnv import environment_array



utility_function = UtilityFunction(kind="ei",xi=0,kappa=0)
def target(x,y,z):
    return environment_array(x,y,z)

optimizer = BayesianOptimization(target, {'z': (2, 4),'y':(40,65),'x':(1000,4000)}, random_state=250)
#optimizer.maximize(init_points=int(input('Enter the number of random steps: ')),n_iter=0)




'''probe_point(np.min(x),np.max(y))
probe_point(np.max(x),np.max(y))
probe_point(np.max(x),np.min(y))'''

def probe_point(x,y,z):
    return optimizer.probe(params={"x": x, "y": y, "z": z},lazy=True,)
probe_point(1000,40,2)
probe_point(1000,40,4)
probe_point(4000,40,4)
probe_point(4000,65,4)
probe_point(4000,65,2)
probe_point(1000,65,2)
probe_point(1000,65,4)
probe_point(2500,52.5,3)
optimizer.maximize(n_iter=0,init_points=0)
optimizer.maximize(n_iter=20, init_points = 2)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
a = []
b = []
j = []
cv = []
def utilitytarget(xtarget,ytarget,ztarget):
    xyparam = np.array([[xtarget,ytarget,ztarget]])
    return float((utility_function.utility(xyparam, optimizer._gp, 0)))






for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = 40
    z = np.random.uniform(2,4)
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1




for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 4
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 2
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

for i in range(10000):

    x = 1000
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

for i in range(10000):

    x = 4000
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1



for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = 65
    z = np.random.uniform(2,4)
    c = utilitytarget(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1





img = ax.scatter(a, b, j,c=cv, cmap=plt.jet(),marker="s")
fig.colorbar(img)
ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')
plt.show()
plt.savefig('3D utility 24 opt steps 3 random')
