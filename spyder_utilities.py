"""
Daniel Maidment

Tue Jun  4 11:20:02 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

########################################################################
############################ Useful Imports ############################
########################################################################

#import os
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import matplotlib
#import time
#from datetime import datetime
#import scipy.io
#import scipy.signal as signal
#import scipy.linalg as linalg
#import warnings
#from numpy import pi
#from numpy import cos
#from numpy import sin
#from numpy import log10
#from numpy import log
#from numpy import real
#from numpy import imag
#from numpy import sqrt
#from numpy import e
#from numpy import abs
#from numpy import angle
#from numpy.fft import fft
#from numpy.fft import ifft
#from numpy.fft import fftshift
#from numpy import min
#from numpy import max
#from numpy import sum
#from numpy.linalg import norm
#from numpy import dot
#from sklearn.preprocessing import normalize

########################################################################
############################## matplotlib ##############################
########################################################################

#from latex_envs.latex_envs import figcaption
#\Lib\site-packages\matplotlib\mpl-data\stylelib
#plt.style.use('seaborn-paper')
plt.style.use('myPaper_color')
#plt.style.use('myPaper_grey')

#turns off automatic plt.plot display, use plt.show()  to re-enable
plt.ioff()
#Turn off pesky from Future Warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

def config_axis(ax=None,
                x_lim=None, X_0=None, y_lim=None, Y_0=None,
                grd=True, minorgrd=False, mult_x=0.2, mult_y=0.2,
                Eng=True, debug=False):
    """
    This function is used to easily configure axes for matplotlib.
    Args:
            ax [axes]:
                Is the axis to be configured,
                i.e., fig, ax = plt.subplots().
            x_lim [tupple]:
                (x_min, x_max) is the shape of the x_data [inclusive].
            y_lim [tupple]:
                (y_min, y_max) is the shape of the y_data [inclusive].
            X_0 [float or int]:
                Is the maximum value of x that one wishes the axis ticks
                to be multiples of, i.e., if x_max is not a clean value
                then X_0 can be set to some round value.
            Y_0 [float or int]:
                Is the maximum value of y that one wishes the axis ticks
                to be multiples of, i.e., if y_max is not a clean value
                then Y_0 can be set to some round value.
            grd [bool]:
                Turns the grid on or off.
            minorgrd [bool]:
                Turns the grid on the minor ticks on or off.
            mult_x [int]:
                Set the interval for the major ticks of the x axis as a
                function of X_0.
            mult_y [int]:
                Set the interval for the major ticks of the y axis as a
                function of Y_0.
            Eng [bool]:
                Is a boolean that selects for engineering notation.
        Returns:
            ax [axes]:

    """
    if(ax == None):
        if(debug):
            print("No axis passed attempting to use current axis.")
        ax = plt.gca()
    if(X_0):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(mult_x*X_0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator((mult_x/5)*X_0))
    if(Eng):
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
    if(Y_0):
        ax.yaxis.set_major_locator(ticker.MultipleLocator(mult_y*Y_0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(mult_y/5*Y_0))
    if(grd):
        ax.grid(b=True, which='major', axis='both')
    else:
        ax.grid(b=False, which='major', axis='both')
    if(minorgrd):
        ax.grid(b=True, which='minor', axis='both')
    else:
        ax.grid(b=False, which='minor', axis='both')

    if(x_lim):
        ax.set_xlim(x_lim)
    if(y_lim):
        ax.set_ylim(y_lim)
    return ax


def binary_ax_config(N, ax=None):
    ax = config_axis(ax,
                     x_lim=(0, N),
                     y_lim=(-0.5, 1.5), Y_0=1, mult_y=1,
                     grd=False, Eng=False)
    return ax


def time_str(debug=False):
    """
    Returns:
        [str] yymmdd_hhmmss
    """
    # current date and time
    now = datetime.now()
    if(debug):print("Now: ", now)
    date = str(datetime.date(now))
    if(debug):print("Date: ", date)
    date = date[0:4]+date[5:7]+date[8:10]
    tme = str(datetime.time(now))
    tme = tme[0:2]+tme[3:5]+tme[6:8]

    return date+'_'+ tme

def save_fig(fig=None,
             path="images", image_nm='default',
             eps=False, debug=False):
    """
    Saves the passed figure as a jpeg by default, optionally saves an
    eps, the name will include the timestamp.
    ---Resolution: 300dpi (IEEE)
    ---Papertype: A4 (only in EPS)

    Args:
        fig [matplotlib figure]:
            Passes the figure to be saved.
        path [str]:
            Takes a string for where the figure should be saved,
            defaults to workingdir\images\.
        image_nm [str]:
            Takes the name the file should be saved as
            (without formatting).
        eps [bool]:
            If true saves an eps file and jpeg file, if false, only
            saves a jpeg file (default False).
    Returns:
        None
    """
    if(fig==None):
        if(debug):
            print("No figure defined.\nAttempting to find the current",
                  "working figure.")
        fig = plt.gcf()

    if(path=="images"):
        if(debug):print("No path specified.")
        path = os.getcwd()+'\\images'
        if(debug):
            print("checking if the current working directory",
                  "has an 'images' folder.")
        if(not os.path.isdir(path)):
            if(debug):
                print("No folder found, making an 'images' folder")
            os.mkdir(path)
    time_stamp = time_str()
    filename = path + '\\' + time_stamp+'_'+image_nm
    fig.savefig(filename+'.jpg', dpi = 300)
    if(eps == True):fig.savefig(filename+'.eps', papertype = 'a4')

    return None

########################################################################
############################ sig processing ############################
########################################################################
def decimate(x_arr, factor = 1):
    N = len(x_arr)
    y_arr = []
    for i in np.arange(0, N, factor):
        y_arr.append(x_arr[i])
    return y_arr

def apply_window(arr, win = 'None', beta = 8.6):
    M = len(arr)
    if(win == 'bartlett'): window = np.bartlett(M)
    elif(win == 'hamming'): window = np.hamming(M)
    elif(win == 'hanning'): window = np.hanning(M)
    elif(win == 'kaiser'): window = np.kaiser(M, beta = beta)
    elif(win == 'blackman'): window = np.blackman(M)
    elif(win == 'None'): window = np.ones(M)
    else:
        window = np.ones(M)

    ret_arr = np.multiply(arr, window)
    return ret_arr

def time_str(debug=False):
    """
    Returns:
        [str] yymmdd_hhmmss
    """
    # current date and time
    now = datetime.now()
    if(debug):print("Now: ", now)
    date = str(datetime.date(now))
    if(debug):print("Date: ", date)
    date = date[0:4]+date[5:7]+date[8:10]
    tme = str(datetime.time(now))
    tme = tme[0:2]+tme[3:5]+tme[6:8]

    return date+'_'+ tme

def save_fig(fig=None,
             path=None, image_nm='default',
             timestamp=True,
             eps=False, debug=False):
    """
    Saves the passed figure as a jpeg by default, optionally saves an
    eps, the name will include the timestamp.
    ---Resolution: 300dpi (IEEE)
    ---Papertype: A4 (only in EPS)

    Args:
        fig [matplotlib figure]:
            Passes the figure to be saved.
        path [str]:
            Takes a string for where the figure should be saved,
            defaults to workingdir\images\.
        image_nm [str]:
            Takes the name the file should be saved as
            (without formatting).
        eps [bool]:
            If true saves an eps file and jpeg file, if false, only
            saves a jpeg file (default False).
    Returns:
        None
    """
    if(fig==None):
        if(debug):
            print("No figure defined.\nAttempting to find the current",
                  "working figure.")
        fig = plt.gcf()

    if(not path):
        if(debug):print("No path specified.\n",
                        "Setting the path to an 'images' directory")
        path = os.getcwd()+"\\images"
        if(debug):
            print("checking if the current working directory",
                  "has an 'images' folder.")
        if(not os.path.isdir(path)):
            if(debug):
                print("No folder found, making an 'images' folder")
            os.mkdir(path)
    else:
        if(debug):
            print("path specified as:", path)
        path = os.getcwd() + path
        if(not os.path.isdir(path)):
            if(debug):
                print("No path found, making a path.")
            os.mkdir(path)

    if(timestamp):
        time_stamp = time_str()
        filename = path + '\\' + time_stamp+'_'+image_nm
    else:
        filename = path + '\\' + image_nm

    fig.savefig(filename+'.jpg', dpi = 300)
    if(eps == True):fig.savefig(filename+'.eps', papertype = 'a4')

    return None

if __name__ == "__main__":
    print(time_str(debug=True))

