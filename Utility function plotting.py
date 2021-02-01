from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
from Environment import environment, environment_array
import matplotlib.pyplot as plt
from matplotlib import gridspec

def target(c):
    return environment_array(c)
b = np.linspace(40,65,1000)
c = np.linspace(2, 4, 1000).reshape(-1,1)
y = target(c)
plt.title('Function to be optimised')
plt.xlabel('Clamping pressure (bar)')
plt.ylabel('Lap shear strength (N)')
plt.plot(c, y);
plt.savefig('Function to be optimised')
plt.show()

optimizer = BayesianOptimization(target, {'c': (2, 4)}, random_state=27)

optimizer.maximize(init_points = 2, n_iter = 0, kappa = 5,xi=0)


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["c"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((2, 4))
    axis.set_ylim((None, None))
    axis.set_ylabel('Lap Shear Strength (N)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ei", kappa=5, xi=0.001)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((2, 4))
    acq.set_ylim((0, np.max(utility) + 300))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})
    plt.show()

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

plot_gp(optimizer, c, y)

optimizer.maximize(init_points=0, n_iter=1)
#plot_gp(optimizer, c, y)

utility_function= UtilityFunction(kind = 'ei', kappa = 5, xi = 0.001)
plt.plot(c,utility_function.utility(c,optimizer._gp, 0),label = '1')
print('The max is: '+str(max(utility_function.utility(c,optimizer._gp, 0))))
g = np.array([2.75]).reshape(-1,1)
print('Returning a single point: '+str(float(utility_function.utility(g,optimizer._gp,0))))


optimizer.maximize(init_points=0, n_iter=1)
print('The max is: '+str(max(utility_function.utility(c,optimizer._gp, 0))))
plt.plot(c,utility_function.utility(c,optimizer._gp, 0),label = '2')
#plot_gp(optimizer, c, y)

optimizer.maximize(init_points=0, n_iter=1)
print('The max is: '+str(max(utility_function.utility(c,optimizer._gp, 0))))
plt.plot(c,utility_function.utility(c,optimizer._gp, 0),label = '3')
#plot_gp(optimizer, c, y)

optimizer.maximize(init_points=0, n_iter=1)
print('The max is: '+str(max(utility_function.utility(c,optimizer._gp, 0))))
#plot_gp(optimizer, c, y)
plt.plot(c,utility_function.utility(c,optimizer._gp, 0),label = '4')

optimizer.maximize(init_points=0, n_iter=1)
print('The max is: '+str(max(utility_function.utility(c,optimizer._gp, 0))))
#plot_gp(optimizer, c, y)
#plt.plot(c,utility_function.utility(c,optimizer._gp, 0))
plt.plot(c,utility_function.utility(c,optimizer._gp, 0),label = '5')

utility_function = UtilityFunction(kind= 'ei', kappa = 5, xi=0.001)
plt.legend()
plt.savefig('1D Utility Functions')
plt.show()