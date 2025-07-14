
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import calc_alpha
from  astropy.cosmology import FlatLambdaCDM, z_at_value
import shared_data
def main():
        cosmo = FlatLambdaCDM(H0=69.09, Om0=.3, Tcmb0=2.725)
        fig, ax = shared_data.set_plot_params(nrows=1, ncols=2)
        norm = mpl.colors.LogNorm(vmin=1e4, vmax=4e7)
        ax_alpha=ax[1]
        ax_rc=ax[0]
        
        #set times
        t_1e4 = np.array([.111, .125, .143,.167, .2, .25, .333, .5])
        t_1e5 = np.array([.111, .125, .167, .2, .25, .333, .5])
        t_4e5 = np.array([.25, .333, .50])
        t_8e5 = np.array([.167, .20, .25, .333, .50])
        t_2e6 = np.array([.167, .20, .25, .333, .50])
        t_5e6 = np.array([.143, .167, .20, .25, .333, .50, .75])
        t_1e7 = np.array([.143, .167, .20, .25, .333, .50, .75])
        t_2e7 = np.array([.111, .143, .167, .20, .25, .333, .50, .75])

        #in terms of z
        z_1e4 = 1/t_1e4 - 1
        z_1e5 = 1/t_1e5 - 1
        z_4e5 = 1/t_4e5 - 1
        z_8e5 = 1/t_8e5 - 1
        z_2e6 = 1/t_2e6 - 1
        z_5e6 = 1/t_5e6 - 1
        z_1e7 = 1/t_1e7 - 1
        z_2e7 = 1/t_2e7 - 1

        #set m_out values
        x_1e4 = np.repeat(1e4, len(t_1e4))/0.6909
        x_1e5 = np.repeat(1e5, len(t_1e5))/0.6909
        x_4e5 = np.repeat(4e5, len(t_4e5))/0.6909
        x_8e5 = np.repeat(8e5, len(t_8e5))/0.6909
        x_2e6 = np.repeat(2e6, len(t_2e6))/0.6909
        x_5e6 = np.repeat(5e6, len(t_5e6))/0.6909
        x_1e7 = np.repeat(1e7, len(t_1e7))/0.6909
        x_2e7 = np.repeat(2e7, len(t_2e7))/0.6909

        #get core radii
        rc_1e4 = pd.read_csv('m8m1rcs.txt', header=None)
        rc_1e5 = pd.read_csv('m2rcs.txt', header=None)
        rc_4e5 = pd.read_csv('m4e5rcs.txt', header=None)
        rc_8e5 = pd.read_csv('m8e5rcs.txt', header=None)
        rc_2e6 = pd.read_csv('m2e6rcs.txt', header=None)
        rc_5e6 = pd.read_csv('m5e6rcs.txt', header=None)
        rc_1e7 = pd.read_csv('m1e7rcs.txt', header=None)
        rc_2e7 = pd.read_csv('m2e7rcs.txt', header=None)

        #get alphas
        a_1e4 = pd.read_csv('m8m1slopes.txt', header=None)
        a_1e5 = pd.read_csv('m2slopes.txt', header=None)
        a_4e5 = pd.read_csv('m4e5slopes.txt', header=None)
        a_8e5 = pd.read_csv('m8e5slopes.txt', header=None)
        a_2e6 = pd.read_csv('m2e6slopes.txt', header=None)
        a_5e6 = pd.read_csv('m5e6slopes.txt', header=None)
        a_1e7 = pd.read_csv('m1e7slopes.txt', header=None)
        a_2e7 = pd.read_csv('m2e7slopes.txt', header=None)

        #pairs of alphas
        ya = [a_1e4, a_1e5, a_4e5, a_8e5, a_2e6, a_5e6, a_1e7, a_2e7]
        xa = [x_1e4, x_1e5, x_4e5, x_8e5, x_2e6, x_5e6, x_1e7, x_2e7]
        #za = [t_1e4, t_1e5, t_4e5, t_8e5, t_2e6, t_5e6, t_1e7, t_2e7]
        za = [z_1e4, z_1e5, z_4e5, z_8e5, z_2e6, z_5e6, z_1e7, z_2e7]
        #pairs of rcs
        yrc = [rc_1e4, rc_1e5, rc_4e5, rc_8e5, rc_2e6, rc_5e6, rc_1e7, rc_2e7]
        xrc = xa
        zrc = za
        s=100

        ax_rc.set_xlabel('Redshift')
        ax_rc.margins(x=0, y=0)
        ax_rc.set_xlim(0, 9)
        ax_rc.set_ylim(0, 2.2)
        ax_alpha.axhspan(-1.9, -1.3, color='lightgrey', alpha=0.75)
        #ax_rc.axvline(x=7, color='black', linestyle='dashdot', label='$z=7$')
        ax_rc.axhline(y=0.106, color='grey', linestyle='dashed')
        #ax_rc.fill_between([ax_rc.get_xlim()[0],6.0], 0.11, ax_rc.get_ylim()[1], color='lightgray', alpha=.7)
        ax_rc.set_ylabel(r'$r_{\rm c}~\rm{(kpc)}$')
        #ax_rc.set_xticks([0.1,0.2,0.3,0.4,0.5])
        ax_rc.legend(frameon=False)
        ax_rc.text(0.55, 0.88, r'$M_{\rm halo} = 7.8 \times 10^{7}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax_rc.transAxes, fontsize=17)
        ax_rc.text(0.53, 0.81, r'$M_{*} = 1.4 \times 10^{3}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax_rc.transAxes, fontsize=17)
        #plot the rcs
        for x1, y1, z1 in zip(xrc, yrc, zrc):
            ax_rc.scatter(z1, y1, norm=norm, cmap=cm.plasma, c=x1, edgecolors='black', s=s)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm.plasma), ax=ax, label='Mass Expelled ($M_{\odot}$)')
        
        #plot the alphas
        ax_alpha.set_xlabel('Redshift')
        ax_alpha.margins(x=0, y=0)
        ax_alpha.set_xlim(0, 9)
        ax_alpha.set_ylim(-2.00, 0.25)
        #ax_alpha.axvline(x=7, color='black', linestyle='dashdot', label='$z=7$')
        ax_alpha.axhspan(-1.9, -1.3, color='lightgrey', alpha=0.6) 
        #ax_rc.text(.48, 0.84, r'\textbf{Single-burst}',verticalalignment='bottom', horizontalalignment='right',transform=ax_rc.transAxes, fontsize=20)
        #ax_alpha.text(.48, 0.87, r'\textbf{Single-burst}',verticalalignment='bottom', horizontalalignment='right',transform=ax_alpha.transAxes, fontsize=20)
        ax_alpha.text(.3, 0.1, r'\textbf{NFW}',weight='bold',verticalalignment='bottom', horizontalalignment='right',transform=ax_alpha.transAxes, fontsize=20)
        ax_rc.text(8.7, 0.15, r'\textbf{2.8 $\times$ softening}', color='gray', fontsize=15)
        ax_alpha.set_ylabel(r'$\alpha$')
        #ax_alpha.text(0.55, 0.73, r'$M_{\rm halo} = 7.8 \times 10^{7}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax_alpha.transAxes, fontsize=17)
        #ax_alpha.text(0.53, 0.66, r'$M_{*} = 1.4 \times 10^{3}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax_alpha.transAxes, fontsize=17)
        for x1, y1, z1 in zip(xa, ya, za):
            ax_alpha.scatter(z1, y1, norm=norm, cmap=cm.plasma, c=x1, edgecolors='black', s=s)
        newlabel = [8, 7, 6, 5, 4, 3, 2, 1]
        #convert = lambda x : cosmo.lookback_time(x).value
        #newpos = [convert(z) for z in newlabel]
        ax_alpha.set_xticks(newlabel)
        ax_alpha.set_xticklabels(newlabel)
        ax_alpha.invert_xaxis()
        ax_rc.set_xticks(newlabel)
        ax_rc.set_xticklabels(newlabel)
        ax_rc.invert_xaxis()

        fig.savefig('sb.pdf', bbox_inches='tight')

        return


if __name__=='__main__':
	main()
 
