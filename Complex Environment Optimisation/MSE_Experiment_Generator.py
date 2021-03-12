import numpy as np

xtest = []
ytest = []
ztest = []

for p in range(100):
    xtest.append(np.random.uniform(1000,4000))
    ytest.append(np.random.uniform(40,65))
    ztest.append(np.random.uniform(2,4))

experiments = np.array(list(zip(xtest,ytest,ztest)))
print(experiments)
np.savetxt('Experiments for MSE.csv', experiments,delimiter = ',')