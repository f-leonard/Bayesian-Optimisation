from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
from TwoDimEnvironmentbc import environment_array
import matplotlib.pyplot as plt

def black_box_function(x, y):
    return environment_array(x,y)

pbounds = {'x': (40, 65), 'y': (2, 4)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=112
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)
xlist = []
ylist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))


x = np.linspace(40,65,100)
y = np.linspace(2,4,100)

x_1, y_1 = np.meshgrid(y,x)
f = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        f[i][j]=environment_array(x[i], y[j])

plt.contourf(x_1, y_1, f)

plt.colorbar()

for i in range(12):

    plt.annotate('x'+str(i+1),(ylist[i],(xlist[i])))
print(np.max(f))
plt.xlim((2,4))
plt.ylim((40,65))
plt.xlabel('Clamping Pressure (Bar)')
plt.ylabel('Vibration Amplitude (micrometres)')


plt.show()


plt.savefig('2D function for optimisation')


