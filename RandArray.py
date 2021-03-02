import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bayesian_optimization import BayesianOptimization


ten= np.sort(np.random.uniform(1500,1900,10000))
twenty = np.flip(np.sort(np.random.uniform(1900,1700,10000)))
thirty = np.sort(np.random.uniform(1700,2250,10000))
forty = np.flip(np.sort(np.random.uniform(2250,1600,10000)))
fifty = np.sort(np.random.uniform(1600,3200,10000))
sixty = np.flip(np.sort(np.random.uniform(3200,3100,10000)))
seventy = np.flip(np.sort(np.random.uniform(3100,800,10000)))
eighty = np.sort(np.random.uniform(800,1000,10000))
ninety = np.sort(np.random.uniform(1000,1320, 10000))
onehundred = np.flip(np.sort(np.random.uniform(1320, 200,10000)))
outputs = np.concatenate((ten,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety,onehundred))
noise = np.random.normal(0,100,100000)
outputs =outputs +noise


i = 0
xarr = []
yarr = []
zarr = []

'''while i <1000:
    x = np.random.uniform(1000,4000)
    y = np.random.uniform(40,65)
    z = np.random.uniform(2,4)

    xarr.append(x)
    yarr.append(y)
    zarr.append(z)

    i = i +1'''
longarray = []
alist = []
blist = []
def makearray():
    for i in range(1000):
        a = np.random.uniform(1000,4000)
        b = np.random.uniform(1000,4000)
        alist.append(float(a))
        blist.append(float(b))
makearray()

def makelongarray():
    i = 0
    while i <len(alist):

        if alist[i]<blist[i]:
            return longarray.append(np.sort(np.random.uniform(alist[i],blist[i],10)))
        else:
            return longarray.append(np.flip(np.sort(np.random.uniform(alist[i],blist[i],10))))

    print(i)

makelongarray()
print(alist)
print(blist)
print(longarray)



'''    
xten= np.sort(np.random.uniform(40,50,1000))
xtwenty = np.sort(np.sort(np.random.uniform(50,52,1000)))
xthirty = np.sort(np.random.uniform(52,52.5,1000))
xforty = np.flip(np.sort(np.random.uniform(52.5,47,1000)))
xfifty = np.sort(np.random.uniform(47,49,1000))
xsixty = np.sort(np.random.uniform(49,60,1000))
xseventy = np.flip(np.sort(np.random.uniform(60,48,1000)))
xeighty = np.sort(np.random.uniform(48,49,1000))
xninety = np.flip(np.sort(np.random.uniform(49,42, 1000)))
xonehundred = np.sort(np.sort(np.random.uniform(42, 54,1000)))
xinputs = np.concatenate((xten,xtwenty,xthirty,xforty,xfifty,xsixty,xseventy,xeighty,xninety,xonehundred))

yten= np.flip(np.sort(np.random.uniform(3870,2459,1000)))
ytwenty = np.flip(np.sort(np.random.uniform(2459,1000,1000)))
ythirty = np.sort(np.random.uniform(1000,2435,1000))
yforty = np.sort(np.sort(np.random.uniform(2435,2500,1000)))
yfifty = np.flip(np.sort(np.random.uniform(2500,2200,1000)))
ysixty = np.sort(np.random.uniform(2200,2412,1000))
yseventy = np.flip(np.sort(np.random.uniform(2412,2000,1000)))
yeighty = np.sort(np.random.uniform(2000,4000,1000))
yninety = np.flip(np.sort(np.random.uniform(4000,2900, 1000)))
yonehundred = np.sort(np.sort(np.random.uniform(2900,2870,1000)))
yinputs = np.concatenate((yten,ytwenty,ythirty,yforty,yfifty,ysixty,yseventy,yeighty,yninety,yonehundred))

zten= np.flip(np.sort(np.random.uniform(2.4,2.389,1000)))
ztwenty = np.sort(np.random.uniform(2.389,2.4,1000))
zthirty = np.flip(np.sort(np.random.uniform(2.4,2.128,1000)))
zforty = np.sort(np.sort(np.random.uniform(2.128,3.928,1000)))
zfifty = np.sort(np.random.uniform(3.928,3.958,1000))
zsixty = np.sort(np.random.uniform(3.958,4,1000))
zseventy = np.flip(np.sort(np.random.uniform(4,2.098,1000)))
zeighty = np.flip(np.sort(np.random.uniform(2.098,2,1000)))
zninety = np.sort(np.random.uniform(2,3.476, 1000))
zonehundred = np.sort(np.sort(np.random.uniform(3.476,3.492,1000)))
zinputs = np.concatenate((zten,ztwenty,zthirty,zforty,zfifty,zsixty,zseventy,zeighty,zninety,zonehundred))'''


'''yarr = np.sort(yarr)
zarr = np.sort(zarr)
inputs = np.array(list(zip(yinputs,xinputs,zinputs)))
inputs_outputs = np.array(list(zip(yinputs,xinputs,zinputs, outputs)))

np.savetxt('Inputs_outputs.csv', inputs_outputs, delimiter=',')'''

