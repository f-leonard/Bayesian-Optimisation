from Environment import environment
import random
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
data=[]
pre_normalisation_adata=[]
adata=[]
for i in range(100):
    a=random.randint(1000, 4000)
    pre_normalisation_adata.append(a)
    a = (a - 1000) / 2500
    adata.append(float(a))
    func=environment(a)
    data.append(float(func))

plt.ylabel('Lap Shear Strength (N)')
plt.xlabel('Welding Energy (''J'')')
plt.plot(pre_normalisation_adata,data)

plt.show()



