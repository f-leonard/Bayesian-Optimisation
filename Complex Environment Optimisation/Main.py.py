from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization,UtilityFunction
from sklearn.gaussian_process.kernels import Matern

'''Read the fabricated experimental data from the csv file and interpolate the environment over a 5x5x5 matrix'''

datacsv = pd.read_csv('Inputs_outputs.csv')
data = datacsv.iloc[:, 0].values
data = data[0:1000]
data = data[1::8]
data = data.reshape(5, 5, 5)

x = np.linspace(10, 40, 5)
y = np.linspace(40, 65, 5)
z = np.linspace(2, 4, 5)



interpolating_function = RegularGridInterpolator((x, y, z), data)


def function(x,y,z):
    '''This function takes a number of type float for the 3 input variables and returns their LSS
    it is used for the experimentation'''
    pts = np.array([[x, y, z]])

    return float(interpolating_function(pts))

utility_function = UtilityFunction(kind="ei",xi=1,kappa=0)

def utilitytarget(xtarget,ytarget,ztarget):
    xyparam = np.array([[xtarget,ytarget,ztarget]])
    return float((utility_function.utility(xyparam, optimizer._gp, 0)))


a = []
b = []
j = []
cv = []
ucv = []

def slicex(x):
    '''Generates requested planes at a specific location on the x axis'''
    for i in range(10000):
        x = x
        y = np.random.uniform(40, 65)
        z = np.random.uniform(2, 4)
        c = function(x, y, z)
        d = utilitytarget(x,y,z)
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        ucv.append(d)
def slicey(y):
    '''Generates requested planes at a specific location on the y axis'''
    for i in range(10000):
        x = np.random.uniform(10,40)
        y = y
        z = np.random.uniform(2, 4)
        c = function(x, y, z)
        d = utilitytarget(x,y,z)
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        ucv.append(d)

def slicez(z):
    '''Generates requested planes at a specific location on the z axis'''
    for i in range(10000):
        x = np.random.uniform(10,40)
        y = np.random.uniform(40, 65)
        z = z
        c = function(x, y, z)
        d = utilitytarget(x,y,z)
        a.append(x*100)
        b.append(y)
        j.append(z)
        cv.append(c)
        ucv.append(d)


'''Instantiates the bounded region of parameter space NB: Welding energy is scaled down by a factor of 100'''
pbounds = {'y': (40, 65), 'z': (2, 4),'x':(10,40)}

'''optimizer calls the BayesianOptimization class, defines the black box function as function from above and takes
the bounded region of parameter space pbounds from above. random_state is a variable which can be specified to make each
set of random experiments repeatable. The results presented in the report can be obtained by setting random_state = 2
and random_state = 1 respectively'''

optimizer = BayesianOptimization(
    f=function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=2

)
def probe_point(x,y,z):
    '''This is a function that allows the user to probe specific points in the environment
    and add them to the environment observations this can be used to guide the optimisation process'''

    return optimizer.probe(params={"x": x, "y": y, "z":z},lazy=True,)

'''The alpha parameter controls how the environment deals with noise. This can be increased in cases where the environment
is more complex. Its base value is 1e-5'''
alpha = 3e-4

'''The Matern kernel is discussed in section 5.2 of the report. nu is the kernel hyperparameter.'''
kernel = Matern(nu = 2.5)

'''Here the structure of the optimizer is initialised. init_points specifies how many steps of random exploration
are requested, n_iter specifies how many steps of BO are requested and xi and kappa are the parameters that control
 the exploration exploitation tradeoff. Xi=1 is tuned for maximum exploration Xi=0 is tuned for maximum exploitation.
 Kappa is the parameter used for the exploration exploitation tradeoff in the ucb acquisition function. kappa = 10 
 corresponds with maximum exploration and kappa = 0 corresponds with maximum exploitation.
'''
optimizer.maximize(init_points = 5, n_iter =5 ,acq = 'ei',kappa=10,xi = 0.4,alpha = alpha,normalize_y = True, kernel= kernel)
optimizer.maximize(init_points =0, n_iter = 5,acq = 'poi',kappa=0,xi = 0.5 ,alpha = alpha,normalize_y = True,kernel = kernel)

xlist = []
ylist = []
zlist = []
datalist = []
'''In this section the optimisation points are retrieved and stored in arrays. The maximum global observation is returned
and its corresponding inputs are passed into the slicex, slicey and slicez functions above to return planes through the
maximum point'''
for res in enumerate(optimizer.res):
    xlist.append(float(res[1].get('params').get('x')))
    ylist.append(float(res[1].get('params').get('y')))
    zlist.append(float(res[1].get('params').get('z')))
    datalist.append(float(res[1].get('target')))

#img = ax.scatter(xlist,ylist,zlist,marker='o',c='black')
datalist = np.array(datalist)
max_element = np.where(datalist == np.amax(datalist))

max_index = int(max_element[0])
slicex(xlist[max_index])
slicey(ylist[max_index])
slicez(zlist[max_index])

def variableacq():
    '''Function available for future work experimenting with the dynamic updating of the acquisition functions
    this function varies the functions linearly from maximum exploration to maximum exploitation '''

    EIsteps = int(input('Enter the number of EI steps requested: '))
    POIsteps = int(input('Enter the number of POI steps requested: '))

    for k in range(EIsteps):
        kappa = 10 - (10 / EIsteps) * k
        xi = 0

        optimizer.maximize(init_points=5, n_iter=5, acq='ei', kappa=kappa, xi=xi, alpha=alpha, normalize_y=True,
                           kernel=kernel)
    for j in range(POIsteps):
        xi = 1 - ((1 / POIsteps) * j)
        optimizer.maximize(init_points=0, n_iter=5, acq='poi', kappa=kappa, xi=xi, alpha=alpha, normalize_y=True,
                           kernel=kernel)

'''The experiments conducted in the BO process are stored in a csv file so they can be read in by other files.'''
optpointsarray = np.array((list((zip(xlist,ylist,zlist)))))
np.savetxt('Optpoints.csv',optpointsarray,delimiter=',')

'''An interactive plot of the function and utility function is generated and structured here. The plots are sized for
 full screen viewing'''
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection = '3d')
img = ax.scatter(a,b,j,c=cv, cmap=plt.jet())
img1 = ax1.scatter(a,b,j,c =ucv, cmap = plt.jet())
ax.view_init(elev=13., azim=-140)
ax1.view_init(elev=13., azim=-140)
fig.figsize = (20,20)
print('The maximum value observed by bayesian optimisation was', np.max(datalist))
ax.set_xlabel('Welding Energy (J)',fontsize = 14)
ax.set_ylabel('Vibration amplitude ('r'$\mu$m)',fontsize = 14)
ax.set_zlabel('Clamping pressure (bar)',fontsize = 14)
ax1.set_xlabel('Welding Energy (J)',fontsize = 14)
ax1.set_ylabel('Vibration amplitude ('r'$\mu$m)',fontsize = 14)
ax1.set_zlabel('Clamping pressure (bar)',fontsize = 14)

cbar1 = fig.colorbar(img,ax =ax)

cbar2 = fig.colorbar(img1, ax = ax1)
cbar1.set_label('LSS (N)', labelpad = -40, y = 1.05, rotation = 0)
cbar2.set_label('LSS (N)', labelpad = -40, y = 1.05, rotation = 0)
plt.suptitle('Acquisition Function and Environment Intersected at Predicted Maximum',fontsize = 16)
plt.show()




