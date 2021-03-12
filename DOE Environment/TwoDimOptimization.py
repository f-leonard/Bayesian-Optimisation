from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
from TwoDimEnvironmentbc import environment_array
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec


def black_box_function(x, y):
    return environment_array(x,y)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

x = np.linspace(40,65,100)
y = np.linspace(2,4,100)

pbounds = {'x': (40, 65), 'y': (2, 4)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=112
)
def probe_point(x,y):
    return optimizer.probe(params={"x": x, "y": y},lazy=True,)

probe_point(40,4)
probe_point(65,4)
probe_point(65,2)
probe_point(52.5,3)
optimizer.maximize(n_iter=0, init_points = 0)


optimizer.maximize(
    init_points=0,
    n_iter=10,
)




xlist = []
ylist = []
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))




x_1, y_1 = np.meshgrid(y,x)
f = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        f[i][j]=environment_array(x[i], y[j])


plt.contourf(x_1, y_1, f)
plt.figure(1)
fig = plt.figure(1)
y = xlist
x = ylist
graph, = plt.plot([], [], 'r+')
def animate(i):
    graph.set_data(x[:i+1], y[:i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=12, interval=2000)

plt.colorbar()




plt.xlabel('Clamping Pressure (Bar)')
plt.ylabel('Vibration Amplitude (micrometres)')

ani.save('GIF of Process.gif', writer = 'imagemagick', fps = 1 )

plt.show()






