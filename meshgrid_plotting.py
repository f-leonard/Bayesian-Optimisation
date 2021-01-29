from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
from numpy import random
from TwoDimEnvironmentbc import environment_array
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

y =[49.37642822183213, 63.75041448512918, 63.74182108956431, 63.468893199480746, 61.9192471339716, 59.0870180481625, 63.010930729949195, 40.0, 55.438232487497494, 65.0, 65.0, 65.0]
x =[3.2806092398452886, 2.151354409404123, 2.192944642094836, 3.514346535937212, 2.8153741854494023, 4.0, 2.853920731192261, 2.0, 2.0, 4.0, 3.257801254759964, 2.8753458272461616]
#x = [0,1,2,3,4,5,6,7,8,9]
#y = [0,2,2,5,4,5,5,7,8,9]
fig = plt.figure()
plt.xlim(2,4)
plt.ylim(40,65)
graph, = plt.plot([], [], 'o')

def animate(i):
    graph.set_data(x[:i+1], y[:i+1])
    return graph

ani = FuncAnimation(fig, animate, frames = 12, interval= 500)


plt.show()