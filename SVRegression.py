import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bayesian_optimization import BayesianOptimization
datasets = pd.read_csv('ExperimentData.csv')
X = datasets.iloc[:, 0:3].values
Y = datasets.iloc[:, 3].values

noise = np.random.normal(0,100,100)

Y = Y+noise

Y = Y.reshape(-1,1)

# Feature Scaling
'''Standardize features by removing mean and scaling it to unit variance'''

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


# Fitting the SVR model to the data
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)


def prediction(x,y,z):
    return float(sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([x,y,z]).reshape(1,-1)))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = []
b = []
j = []
cv = []

for i in range(50000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 2
    c = prediction(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

for i in range(50000):

    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = 4
    c = prediction(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

for i in range(50000):

    x = 2500
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)
    c = prediction(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1
for i in range(50000):

    x = np.random.uniform(1000,4000)
    y = 52.5
    z = np.random.uniform(2,4)
    c = prediction(x,y,z)
    i = i +1
    a.append(x)
    b.append(y)
    j.append(z)
    cv.append(c)
    i = i + 1

pbounds = {'y': (40, 65), 'z': (2, 4),'x':(1000,4000)}
optimizer = BayesianOptimization(
    f=prediction,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=249
)
def probe_point(x,y,z):
    return optimizer.probe(params={"x": x, "y": y, "z":z},lazy=True,)

probe_point(4000,65,4)
probe_point(4000,65,2)
probe_point(1000,65,2)
probe_point(1500,50,3)
optimizer.maximize(init_points = 3, n_iter = 20)

xlist = []
ylist = []
zlist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))
    zlist.append(float(res[1].get('params').get('z')))

ax.scatter(xlist,ylist,zlist,marker='o')

cv = np.array(cv)
img = ax.scatter(a, b, j,c=cv, cmap=plt.jet())
fig.colorbar(img)
print('The maximum value observed on the plot was', max(cv))
ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')

plt.savefig('4D plot of environment')
plt.show()