#Function to Graph ANN Data


import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def graphAnn( X, Y, save=True):
    plt.figure(figsize=(16, 10))
    plt.rc('axes', edgecolor='black', linewidth=2)
    font = font_manager.FontProperties(family='Arial', style='normal', size=20)
    plt.plot((X), linewidth=5, color='r')  # color='r' to change colour to red etc
    plt.plot((Y), linewidth=5, color='b')
    plt.ylabel("Mean Squared Error", fontname='Arial', fontsize='20', fontweight='bold', labelpad=25)
    plt.xlabel("Epoch", fontname='Arial', fontsize='20', fontweight='bold', labelpad=25)
    plt.legend(['Training Data', 'Validation Data'], loc='upper right', prop=font)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', length=5, width=2, labelsize=16)
    plt.tick_params(axis='both', which='major', direction='inout', length=12.5, width=2, labelsize=16)
    plt.xlim(xmin=0)
    plt.title(label="ANN Performance", loc='center', fontname='Arial', fontsize='25', fontweight='bold')
    plt.show()
    if save == True:
        plt.savefig("ANN.png")
    return

