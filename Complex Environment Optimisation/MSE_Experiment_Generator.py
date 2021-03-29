import numpy as np

xtest = []
ytest = []
ztest = []

for p in range(100):
    xtest.append(np.random.uniform(2196.58,2796.58))
    ytest.append(np.random.uniform(50,55))
    ztest.append(np.random.uniform(3.3,3.7))

experiments = np.array(list(zip(xtest,ytest,ztest)))
print(experiments)
np.savetxt('Experiments for optimum region MSE.csv', experiments,delimiter = ',')