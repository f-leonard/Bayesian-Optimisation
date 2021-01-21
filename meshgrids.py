import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
#h = plt.contourf(x,y,z)
#plt.show()

y=np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
#y=np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1) + 2*np.exp(-(x - 5)**2) + np.exp(-(x - 8)**2/10) +np.exp(-(x - 10)**2)
x=np.linspace(-2,12.5,100)
plt.plot(x,y)
plt.show()