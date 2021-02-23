import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bayesian_optimization import BayesianOptimization


ten= np.sort(np.random.uniform(1500,1900,10))
twenty = np.flip(np.sort(np.random.uniform(1900,1700,10)))
thirty = np.sort(np.random.uniform(1700,2250,10))
forty = np.flip(np.sort(np.random.uniform(2250,1600,10)))
fifty = np.sort(np.random.uniform(1600,3200,10))
sixty = np.flip(np.sort(np.random.uniform(3200,3100,10)))
seventy = np.flip(np.sort(np.random.uniform(3100,800,10)))
eighty = np.sort(np.random.uniform(800,1000,10))
ninety = np.sort(np.random.uniform(1000,1320, 10))
onehundred = np.flip(np.sort(np.random.uniform(1320, 200,10)))
outputs = np.concatenate((ten,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety,onehundred))
noise = np.random.normal(0,100,100)
outputs =outputs +noise

i = 0
xarr = []
yarr = []
zarr = []

while i <100:
    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)

    xarr.append(x)
    yarr.append(y)
    zarr.append(z)

    i = i +1

xarr = np.sort(xarr)
yarr = np.sort(yarr)
zarr = np.sort(zarr)
inputs = np.array(list(zip(xarr,yarr,zarr)))
inputs_outputs = np.array(list(zip(xarr,yarr,zarr, outputs)))
print(inputs_outputs)

np.savetxt('Inputs_outputs.csv', inputs_outputs, delimiter=',')

