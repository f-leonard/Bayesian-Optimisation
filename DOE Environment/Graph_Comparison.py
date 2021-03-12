#Function to Create a Comparison Graph

import matplotlib.pyplot as plt
from matplotlib import font_manager


def graphComparison(X ,Y ,X_label ,Y_label ,title ,save=False):
    plt.figure(figsize=(16 ,10))
    plt.rc('axes' ,edgecolor='black' ,linewidth=2)
    font = font_manager.FontProperties(family='Arial', style='normal', size=20)
    plt.plot((X) ,linewidth=2 ,marker='o' ,color='r'  )  # color='r' to change colour to red etc
    plt.plot((Y) ,linewidth=2 ,marker='o' ,color='b')
    plt.ylabel(Y_label ,fontname='Arial' ,fontsize='20' ,fontweight='bold' ,labelpad=25)
    plt.xlabel(X_label ,fontname='Arial' ,fontsize='20' ,fontweight='bold' ,labelpad=25)
    plt.legend(['Predicted Data', 'Measured Data'], loc='upper right' ,prop=font)
    plt.minorticks_on()
    plt.tick_params(axis='both' ,which='minor' ,direction='in' ,length=5 ,width=2 ,labelsize=16)
    plt.tick_params(axis='both' ,which='major' ,direction='inout' ,length=12.5 ,width=2 ,labelsize=16)
    plt.xlim(xmin=0)
    plt.title(label=title, loc='center' ,fontname='Arial' ,fontsize='25' ,fontweight='bold')
    if save==True :
        plt.savefig("ANN.png")
    return