from bayesian_optimization import BayesianOptimization
from TwoDimEnvironmentbc import environment_array
from bayesian_optimization import UtilityFunction


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return environment_array(x,y)

pbounds = {'x': (40,65), 'y': (2,4)}


optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
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
