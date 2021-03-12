import sys
from Environment import environment
sys.path.append('../')
from bayesian_optimization import DiscreteBayesianOptimization, UtilityFunction
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import plotting
data = []

g = []
h = []
j = []


cvalues=np.linspace(2, 4, 10000)
avalues = np.linspace(1000,4000,len(cvalues))
bvalues=np.linspace(40, 65, len(cvalues))
cdata=np.linspace(2, 4, len(cvalues))
bdata=np.linspace(40, 65, len(cvalues))

for i in range(len(bdata)):
    avalues[i] = (bvalues[i]-1000)/2500
    bvalues[i] = (bvalues[i] - 45) / 20
    cvalues[i] = (cvalues[i] - 2) / 2

    g.append(avalues[i])
    h.append(bvalues[i])
    j.append(cvalues[i])

    func=environment(j[i])
    data.append(float(func))


data=list(data)
plt.ylabel('Lap Shear Strength (N)')
plt.xlabel('Clamping pressure (bar)')
plt.plot(cdata,data)
plt.show()
#########################################################################



data=np.round(data)
def target(x):
    return float(environment((x-2)/2))/250


x = np.linspace(2,4,10000).reshape(-1,1)


y = data.reshape(-1,1)/250 #clamping pressure vector
firstsample = float(input("Enter the first sample point: "))
secondsample = float(input("Enter the second sample point: "))


plt.plot(x,y)
plt.show()
prange = {'x':(firstsample, secondsample,0.1)} #The last constant here determines how the greedy steps operate. The greedy steps start at the global max obtained from the kmbbo steps and step back or forward while the function increases until it begins to decrease.
random_state = 1234 #Optionally specify a seed for a random number generator
sampler = 'KMBBO' #3 options KMBBO Greedy and capitalist
kwargs = {'multiprocessing':1,
         'n_warmup':10} #n_warmup is the number of times to randomly sample the acquisition function multiprocessing is number of cores for multiprocessing of scipy.minimize
batch_size = 3 #This is the number of observation points in each step

KMBBO_steps=4
greedy_steps=2
# Initialize optimizer and utility function
dbo = DiscreteBayesianOptimization(f=None,
                                  prange=prange,
                                  random_state=random_state)
#if UCB is to be used a constant value of kappa is required. UCB returns mean + kappa * std
#if ei is used z = (mean - y_max - xi) / std and it returns (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
#if POI is used z = (mean - y_max - xi) / std  and it returns norm.cdf(z)
utility = UtilityFunction(kind='ucb', kappa=1, xi= 5)

########################################################################
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

ylimit1= float(input("Enter the first y-axis limit: "))
ylimit2= float(input("Enter the second y-axis limit: "))
################################################################################################################################
#Potential Plotting funciton def plot_gp_1d(gp, df, axis, vector=None, utility_function=None, path=None, dpi=300, threshold=0.6):
#gp = gaussian process
#df = df: dataframe of X data with labels, and y data labeled Target
#axis: axis of reference by string or index
#vector: If given, must correspond to indexing of dataframe
#utility_function: instance of UtilityFunction, default to ucb ('greedy') 2.5.
#path: path for plot saving if desired
#dpi: dots per inch for output figure
#threshold: threshold for kernel similarity measure
################################################################################################################################

def plot_gp(optimizer, x, y):#This is a function that plots the optimizer function
    fig = plt.figure(figsize=(16, 16))
    steps = len(optimizer.space)
    #fig.suptitle(
             #'Bayesian optimization of a Gaussian process',
             #fontdict={'size':30}
    #)
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
              alpha=.9, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((firstsample, secondsample))
    axis.set_ylim((ylimit1, ylimit2))
    axis.set_ylabel('y', fontdict={'size': 15})
    axis.set_xlabel('x', fontdict={'size': 15})

    utility_function = UtilityFunction(kind="ucb", kappa=1, xi=5)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='green')
    #     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
    #              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((firstsample,secondsample))
    acq.set_ylim((min(utility),(max(utility))+min(utility)))
    acq.set_ylabel('Utility Function', fontdict={'size': 8})
    acq.set_xlabel('Clamping Pressure(bar)', fontdict={'size': 6})

    axis.legend(loc=1, borderaxespad=0.5)
    acq.legend(loc=1, borderaxespad=0.5)
    plt.savefig('./' + str(steps) + '.png', dpi=300)
    plt.show()


def step(dbo, sampler='KMBBO'):
    batch = dbo.suggest(utility, sampler=sampler, n_acqs=batch_size, fit_gp=True, **kwargs)
    for point in batch:
        dbo.register(params=point, target=target(**point))
    plot_gp(dbo, x, y)
    plt.show()

batch = [{'x':firstsample},{'x':secondsample}]
for point in batch:
    dbo.register(params=point, target=target(**point))
    plot_gp(dbo, x, y)


for _ in range(KMBBO_steps):
    step(dbo)

for _ in range(greedy_steps):
    step(dbo, 'greedy')