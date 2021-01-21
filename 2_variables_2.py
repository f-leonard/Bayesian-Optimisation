
import numpy as np


from Environment import environment
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
data = []

j = []

cvalues=np.linspace(2, 4, 100)
bvalues=np.linspace(40, 65, len(cvalues))
cdata=np.linspace(2, 4, len(cvalues))
bdata=np.linspace(40, 65, len(cvalues))

for i in range(len(bdata)):

    cvalues[i] = (cvalues[i] - 2) / 2

    j.append(cvalues[i])
    func=environment(j[i])
    data.append(float(func))


data=list(data)
plt.ylabel('Lap Shear Strength (N)')
plt.xlabel('Clamping pressure (bar)')
plt.plot(cdata,data)
plt.show()
#########################################################################
import sys
sys.path.append('../')
from bayesian_optimization import DiscreteBayesianOptimization, UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


data=np.round(data)


def target(x):
    return float(environment((x-2)/2))

x = np.linspace(2,4,100).reshape(-1,1)
y = data


##plt.plot(,y)
#plt.show()
prange = {'x':(2, 4 ,0.01)}
random_state = 1234
sampler = 'KMBBO'
kwargs = {'multiprocessing':1,
         'n_warmup':5}
batch_size = 1

KMBBO_steps=1
greedy_steps=2
# Initialize optimizer and utility function
dbo = DiscreteBayesianOptimization(f=None,
                                  prange=prange,
                                  random_state=random_state)
utility = UtilityFunction(kind='ei', kappa=0.1, xi=0.5)

########################################################################
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 16))
    steps = len(optimizer.space)
    #     fig.suptitle(
    #         'Bayesian optimization of a Gaussian process',
    #         fontdict={'size':30}
    #     )
    plt.rcParams['font.sans-serif'] = "DejaVu Sans"
    plt.rcParams['font.family'] = "sans-serif"
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Activity')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=3, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((2, 4))
    axis.set_ylim((2700,3000))
    axis.set_ylabel('y', fontdict={'size': 15})
    axis.set_xlabel('x', fontdict={'size': 15})

    utility_function = UtilityFunction(kind="ei", kappa=0.1, xi=0.1)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='green')
    #     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
    #              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((2, 4))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Expected improvement', fontdict={'size': 8})
    acq.set_xlabel('x', fontdict={'size': 8})

    axis.legend(loc=1, borderaxespad=0.5)
    acq.legend(loc=1, borderaxespad=0.5)
    plt.savefig('./' + str(steps) + '.png', dpi=300)
    plt.show()


def step(dbo, sampler='KMBBO'):
    batch = dbo.suggest(utility, sampler=sampler, n_acqs=batch_size, fit_gp=True, **kwargs)
    for point in batch:
        dbo.register(params=point, target=target(**point))
    plot_gp(dbo, x, y)

batch = [{'x': 2},{'x':4}]
for point in batch:
    dbo.register(params=point, target=target(**point))
    plot_gp(dbo, x, y)

for _ in range(KMBBO_steps):
    step(dbo)

for _ in range(greedy_steps):
    step(dbo, 'greedy')