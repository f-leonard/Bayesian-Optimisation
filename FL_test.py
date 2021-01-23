#commit test
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization, UtilityFunction, DiscreteBayesianOptimization
from Environment import environment, environment_array

data = []

g = []
h = []
j = []


cvalues=np.linspace(2,4,1000)
avalues = np.linspace(1000,1500,len(cvalues))
bvalues=np.linspace(40, 65, len(cvalues))
cdata=np.linspace(2, 4, len(cvalues))
bdata=np.linspace(40, 65, len(cvalues))

for i in range(len(bdata)):
    avalues[i] = (avalues[i]-1000)/2500
    bvalues[i] = (bvalues[i] - 45) / 20
    cvalues[i] = (cvalues[i] - 2) / 2

    g.append(avalues[i])
    h.append(bvalues[i])
    j.append(cvalues[i])


    data.append(float(environment(j[i])))
    data=list(data)

plt.ylabel('Lap Shear Strength (N)')
plt.xlabel('Clamping pressure (bar)')
data=np.round(data)
plt.plot(cdata,data)
plt.show()


plt.show()
b = bvalues
c = cvalues
np.random.seed(42)


def f(x):
    return environment_array(x)
'''A minimum of 2 initial guesses is required to kickstart the optimization process these may be
 random or user defined.'''
optimizer = BayesianOptimization(f, {'x':(2,4)},random_state=27)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs,y_obs)
    mu,sigma = optimizer._gp.predict(grid, return_std = True)
    return mu, sigma

def plot_bo(f, bo):
    
    x = np.linspace(2, 4, 1000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    #error arrising here determining distance between each pair of two collections of inputs
    plt.figure(figsize=(16, 10))
    plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.xlabel('Clamping pressure (bar)')
    plt.ylabel('Lap Shear Strength(N)')
    plt.savefig(fname = input('Enter plot name: '))
    plt.show()

#ucb
#Prefer exploitation kappa = 1
#Prefer exploration (kappa = 10)

bo = BayesianOptimization(
    f=f,
    pbounds={'x':(2,4)},
    verbose=2,
    random_state=4,
)

bo.maximize(n_iter=4, acq="ucb", xi= 0, kappa=1, init_points=3, alpha= 1e-3)

plot_bo(f, bo)


#ei prefer exploration xi = 0

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (2, 4)},
    verbose=2,
    random_state=4,
)

bo.maximize(n_iter=4, acq="ei", xi=0.01, init_points = 3)

plot_bo(f, bo)

#same process for poi exploitation xi = 0 exploration xi=0.1
bo = BayesianOptimization(
    f=f,
    pbounds={"x": (2, 4)},
    verbose=2,
    random_state=4,
)

bo.maximize(n_iter=4, acq="poi", xi=0.0001, init_points = 3)

plot_bo(f, bo)