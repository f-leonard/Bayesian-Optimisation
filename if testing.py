from Environment import environment, environment_array
import numpy as np

def decision(x):
        return float(environment((x-2)/2)) if type(x) == float or int else environment_array((x-2)/2)
x=np.linspace(2,4,100)

print(decision(x))