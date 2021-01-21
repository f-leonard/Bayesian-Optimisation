#Function to Create a Linear Regression Graph

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas
from scipy import stats


def graphLinearRegression( X, Y, X_label, Y_label, title, save=True):
    plt.figure(figsize=(16, 10))
    plt.rc('axes', edgecolor='black', linewidth=2)
    plt.ylabel(Y_label, fontname='Arial', fontsize='20', fontweight='bold', labelpad=25)
    plt.xlabel(X_label, fontname='Arial', fontsize='20', fontweight='bold', labelpad=25)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', length=5, width=2, labelsize=16)
    plt.tick_params(axis='both', which='major', direction='inout', length=12.5, width=2, labelsize=16)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions plt.scatter(X, Y)
    X = X.astype(float)
    Y = Y.astype(float)
    if type(Y) == pandas.core.frame.DataFrame:
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.iloc[:, 0], Y.iloc[:, 0])
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.iloc[:, 0], Y[:, 0])
    #np.cov(r_value.astype(float), rowvar=False)
    plt.text(0, 0, 'r = ' + str(r_value), fontname='Arial', fontsize=20, fontweight='bold')
    plt.title(label="Linear Regression", loc='center', fontname='Arial', fontsize='25', fontweight='bold')
    plt.scatter(X, Y)
    plt.plot(X.iloc[:, 0], Y_pred, color='red')
    plt.title(label=title, loc='center', fontname='Arial', fontsize='25', fontweight='bold')
    if save == True:
        plt.savefig("Linear Regression.png")
    return