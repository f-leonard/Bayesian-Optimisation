from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from ComplexEnv import environment_array
from bayesian_optimization import BayesianOptimization

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
a = []
b = []
j = []
cv = []

def black_box_function(x, y,z):
    return environment_array(x,y,z)

'''for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 2
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1'''



'''for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 4
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1'''

for i in range(10000):

    x = 1000
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    c = environment_array(x,y,z)
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
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

'''for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = 40
    z = np.random.uniform(2,4)
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1'''

for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = 52.5
    z = np.random.uniform(2,4)
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1



'''for i in range(10000):

    x = np.random.uniform(1000,4000)
    y = 65
    z = np.random.uniform(2,4)
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1'''

for i in range(10000):

    x = 2500
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    c = environment_array(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1
cv = np.array(cv)
'''Here is where the gaussian noise is created and added to the plot'''
'''noise = np.random.normal(0,100,cv.shape)
cv = cv +noise'''
img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
fig.colorbar(img)
print(max(cv))

pbounds = {'y': (40, 65), 'z': (2, 4),'x':(1000,4000)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=249
)
'''def probe_point(x,y,z):
    return optimizer.probe(params={"x": x, "y": y, "z":z},lazy=True,)

probe_point(4000,65,4)
probe_point(4000,65,2)
probe_point(1000,65,2)
probe_point(1500,50,3)'''
optimizer.maximize(init_points = 3, n_iter = 20)

xlist = []
ylist = []
zlist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))
    zlist.append(float(res[1].get('params').get('z')))

ax.scatter(xlist,ylist,zlist,marker='+')

ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')

plt.savefig('4D plot of environment')
plt.show()
