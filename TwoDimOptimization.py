from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
from TwoDimEnvironmentbc import environment_array
import matplotlib.pyplot as plt


f = np.zeros((100,100))
c = np.linspace(2,4,100)
b = np.linspace(40,65,100)
for i in range(100):
    for j in range(100):
        f[i][j]=environment_array(b[i], c[j])


print(f)
k = []
while i < 3000:
    k.append(i)
    i = i +100


plt.contourf(f,levels = k)
plt.xlabel('Clamping Pressure (Bar)')
plt.ylabel('Vibration Amplitude um')
plt.xticks(np.arange(0,125,step=25), [2,2.5,3,3.5,4])
plt.yticks(list(np.linspace(0,100,6)),([40,45,50,55,60,65]))
plt.colorbar()
plt.title('Function to be optimised')
plt.savefig('2D function for optimisation')
#plt.show()
print(np.max(f))

def black_box_function(x, y):
    return environment_array(x,y)

pbounds = {'x': (45, 65), 'y': (2.2, 3.8)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)
x = []
y = []
for res in enumerate(optimizer.res):
    x.append(float(res[1].get('params').get('x')))
    y.append(float(res[1].get('params').get('y')))
print(x)
print(y)

