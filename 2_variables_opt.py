import sys
sys.path.append('../')
from Environment import environment
from bayesian_optimization import DiscreteBayesianOptimization, UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def target(a,b):
    return environment(a,b)
