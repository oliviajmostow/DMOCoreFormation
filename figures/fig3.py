
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from matplotlib.colors import LogNorm
import matplotlib as mpl
import pandas as pd
import calculations
import shared_data
from scipy.optimize import curve_fit

rmin = 0.038
rmax = 10
snap = 127
sphere_samples = 200
radial_samples = 50
DesNgb = 32
h = 0.6909

def main():
        ms = 60
        lw = 3.75
        fig, ax = shared_data.set_plot_params(ncols=2)
        rc_ax = ax[1]
        a_ax = ax[0]
        #core radius for no feedback
        rc_ax.scatter(0.01, 0.1263, s=ms, color='black', label='Smooth model')
        #ax.axhline(y=0.15984, color='black', label='No burst')
        #core radii for increasing energy
        bns = np.array([31, 20, 10, 5,1])
        bns = bns[::-1]
        #need to edit this line
        rcs = pd.read_csv('ups8rcs_min038.txt', header=None)[0]
        rc_ax.scatter(bns, rcs, color='blue', s=ms)
        rc_ax.plot(bns, rcs, linewidth=lw, color='blue', alpha=0.15, ls='--', label='Variable total mass')
        rc_ax.set_xlabel('Number of Bursts')
        rc_ax.set_ylabel(r'${r_{\rm c}}~$(kpc)')
        
        #add fixed energy values in red
        bns2 = np.array([62, 31, 20, 10, 5, 3, 2, 1])
        rcs2 = pd.read_csv('upfmrcs_min038.txt', header=None)[0][0:8]
        rc_ax.scatter(bns2[0:6], rcs2[0:6], color='red', s=ms)
        rc_ax.scatter(bns2[7:], rcs2[7:], color='red', s=ms)
        rc_ax.scatter(bns2[6], rcs2[6], color='red', marker='x', s=ms)
        lbns = bns2[np.arange(len(bns2))!=6]
        lrcs = rcs2[np.arange(len(rcs2))!=6]
        rc_ax.plot(lbns, lrcs, color='red', linewidth=lw,  alpha=0.15, label='Fixed total mass')

        
        rc_ax.set_xlim(-1.0,65)
        rc_ax.set_xticks(np.arange(0,70,10))
       
        ##with slope
        a_ax.scatter(0.01, -1.40, s=ms, color='black', label='Smooth model')
        rcs = pd.read_csv('ups8slope.txt', header=None)[0]
        a_ax.scatter(bns, rcs, color='blue', s=ms)
        a_ax.plot(bns, rcs, color='blue', alpha=0.15, linewidth=lw, ls='--', label='Variable total mass')
        a_ax.set_xlabel('Number of Bursts')
        a_ax.set_ylabel(r'$\alpha$')
        rcs2 = pd.read_csv('upfmslope.txt', header=None)[0][0:8]
        a_ax.scatter(bns2, rcs2, color='red', s=ms)
        a_ax.plot(bns2, rcs2, color='red', alpha=0.15, linewidth=lw, label='Fixed total mass')
        a_ax.set_xlim(-1.0,65)
        a_ax.set_xticks(np.arange(0,70,10))
        a_ax.text(0.95, 0.52, r'$M_{\rm halo} = 9.2 \times 10^{9}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=a_ax.transAxes, fontsize=15)
        a_ax.text(0.95, 0.59, r'$M_{*} = 3.9 \times 10^{6}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=a_ax.transAxes, fontsize=15)
        rc_ax.text(0.95, 0.52, r'$M_{\rm halo} = 9.2 \times 10^{9}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=rc_ax.transAxes, fontsize=15)
        rc_ax.text(0.95, 0.59, r'$M_{*} = 3.9 \times 10^{6}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=rc_ax.transAxes, fontsize=15)

        a_ax.legend(frameon=False)
        rc_ax.legend(frameon=False)
        fig.savefig('aplot.pdf', bbox_inches='tight')
        return


if __name__=='__main__':
	main()
 
