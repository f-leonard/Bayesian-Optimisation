from bayesian_optimization import UtilityFunction, BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from TwoDimEnvironmentbc import environment_array



utility_function = UtilityFunction(kind="ei",xi=0,kappa=0)
def target(x,y):
    return environment_array(x,y)

optimizer = BayesianOptimization(target, {'y': (2, 4),'x':(40,65)}, random_state=112)
optimizer.maximize(init_points=int(input('Enter the number of random steps: ')),n_iter=0)

x = np.linspace(40,65,100)
y = np.linspace(2,4,100)

x_1, y_1 = np.meshgrid(y,x)

def utilitytarget(xtarget,ytarget):
    xyparam = np.array([[xtarget,ytarget]])
    return float((utility_function.utility(xyparam, optimizer._gp, 0)))

f = np.zeros((100,100))
i = 0
array_list = []
longlist = []
optsteps = int(input('Enter the number of Optimisation steps: '))
while i < optsteps:
    optimizer.maximize(init_points=0, n_iter=1)
    for k in range(100):
        for z in range(100):
            f[k][z] = utilitytarget(x[k], y[z])
            longlist.append(f[k][z])


    '''plt.figure(i+1)
    plt.contourf(x_1, y_1, f)
    plt.xlabel('Clamping pressure (Bar)')
    plt.ylabel('Vibration Amplitude (Micrometres)')
    plt.colorbar()
    #plt.show()'''

    i = i+1

longlist = np.array(longlist)

DATA = longlist.reshape(optsteps,100,100)


fig,ax = plt.subplots()

def animate(i):
       ax.clear()
       ax.contourf(x_1,y_1,DATA[i])
       ax.set_title('Optimisation Step '+'%01d'%(i+1))
       ax.set_xlim(2,4)
       ax.set_ylim(40,65)

interval = 2#in seconds
ani = animation.FuncAnimation(fig,animate,optsteps,interval=interval*1e+3,blit=False)
ani.save('GIF of Utility Function.gif', writer = 'imagemagick', fps = 1 )
plt.show()
