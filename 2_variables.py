#optimisation imports
import sys
sys.path.append('../')
from bayesian_optimization import DiscreteBayesianOptimization, UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


from Environment import environment
import random
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
data = []
pre_normalisation_cdata = []
pre_normalisation_bdata = []
bdata = []
cdata = []

for i in range(100):
    b = random.randint(40, 65)  # Choose a random vibration amplitude value
    c = random.randint(2, 3)
    pre_normalisation_bdata.append(b)
    pre_normalisation_cdata.append(c)
    b = (b - 45) / 20
    c = (c - 2) / 2
    bdata.append(float(b))
    cdata.append(float(b))
    func=environment(b, c)
    data.append(float(func))


data=list(data)
plt.ylabel('Lap Shear Strength (N)')
plt.xlabel('Welding Energy (J) and Vibration Amplitude(um)\nN.B Vibration amplitude ranges between 40 and 65 um and increases with vibration')
plt.plot(sorted(pre_normalisation_bdata),sorted(data))
sorted_avalues=sorted(pre_normalisation_bdata)
sorted_bvalues=sorted(pre_normalisation_cdata)
plt.show()

