import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bayesian_optimization import BayesianOptimization
datasets = pd.read_csv('newcsv.csv')
X = datasets.iloc[:, 0:3].values
Y = datasets.iloc[:, 3].values
print(Y[1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cv= []
a = []
b = []
c = []

for i in range(len(X)):
    a.append(X[i][0])
    b.append(X[i][1])
    c.append(X[i][2])
    cv.append(Y[i])

img = ax.scatter(a, b, c, c=cv, cmap=plt.jet())

ax.set_xlabel('Welding Energy (J)')
ax.set_ylabel('Vibration amplitude(um)')
ax.set_zlabel('Clamping pressure (bar)')
fig.colorbar(img)
plt.show()


