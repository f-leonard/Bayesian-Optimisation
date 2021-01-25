import matplotlib.pyplot as plt
import numpy as np
from TwoDimEnvironmentbc import environment_array
f = np.zeros((100,100))
c = np.linspace(2,4,100)
b = np.linspace(40,65,100)
for i in range(100):
    for j in range(100):
        f[i][j]=environment_array(b[i], c[j])


print(f)
k = []
while i < 3000:
    k.append(i)
    i = i +100


plt.contourf(f,levels = k)
plt.xlabel('Clamping Pressure (Bar)')
plt.ylabel('Welding energy (J)?')
plt.xticks(np.arange(0,125,step=25), [2,2.5,3,3.5,4])
plt.yticks(list(np.linspace(0,100,6)),([40,45,50,55,60,65]))
plt.colorbar()
plt.show()
print(np.max(f))
