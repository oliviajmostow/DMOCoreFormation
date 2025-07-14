from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib, os
from matplotlib import rc
from matplotlib import pyplot as plt

def set_ticks(ax):
    ax.tick_params('both', which='minor', length=4, direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', which='major', length=8, direction='in', bottom=True, top=True, left=True, right=True)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    return 

def set_plot_params(nrows=1, ncols=1, figsize=None):

    plt.rc('font',**{'family':'STIXGeneral'})
    plt.rc('text', usetex=True)

    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)

    plt.rc('lines', linewidth=2)

    if figsize is None:
        fig, ax = plt.subplots(nrows,ncols, figsize=(ncols*5 + (ncols)*3,nrows*5+(nrows-1)*3))
    else:
         fig, ax = plt.subplots(nrows,ncols, figsize=figsize)

    if type(ax)==type(np.zeros(1)):
        for a in ax.ravel():
            set_ticks(a)
    else:
        set_ticks(ax)

    return fig, ax
