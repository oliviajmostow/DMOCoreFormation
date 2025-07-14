
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from matplotlib.colors import LogNorm
import matplotlib as mpl
import pandas as pd
import calculations
import shared_data

rmin = 0.038
rmax = 10
snap = 127
sphere_samples = 200
radial_samples = 50
DesNgb = 32
h = 0.6909


def main():
        fig, ax = shared_data.set_plot_params()
        runs2=['../m10_ic0_sg_713/output', '../updated_125e8out/b62/output', '../updated_125e8out/b31/output','../updated_125e8out/b20/output','../updated_125e8out/b10/output','../updated_m10_713s/s8b5/output','../updated_125e8out/b3/output','../updated_125e8out/b2/output','../updated_125e8out/b1_1/output']
        runs = np.array(runs2)[[0,-4, -3, -2, -1]]
        density_list = []
        all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
        for i in range(len(runs)):
            print(f'Calculating density for run {i}')
            all_density = []
            cat = DataLoader(runs[i], snap, 1, ['Masses','Coordinates','SubhaloPos'], sub_idx=0)
            coords = cat['PartType1/Coordinates']/0.6909 - cat['SubhaloPos']/0.6909
            masses = cat['PartType1/Masses']*1e10/0.6909
            for r in all_r:
                points = calculations.fibonacci_sphere(sphere_samples, r)
                density = calculations.calc_density(coords, masses, points, DesNgb)
                all_density.append(density)
            density_list.append(all_density)
        density_list = np.array(density_list)
        density_sg = density_list[0]
        #colorbar
        norm = mpl.colors.Normalize(vmin=1, vmax=5)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        densities = density_list[1:]
        burst_nums = [62, 31, 20, 10, 5, 3, 2, 1]
        #need to also be changing this line
        burst_nums = [5,3,2,1]
        log_nums = np.log2(np.array(burst_nums))
        Z = [[0,0],[0,0]]
        levels = np.array([1,2,3,5,6])
        mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
        CS3 = plt.contourf(Z, levels, cmap=mymap)
        ax.cla()
        y = densities
        z = burst_nums
        for y1,z1 in zip(y,z):
            r = (float(z1-1))/(4)
            g=0
            b=1-r
            ax.plot(all_r, y1, color=(r,g,b), label=z1)
        #cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mymap), ax=ax, label='Number of Bursts')
        #cbar = fig.colorbar(CS3, ax=ax)
        #cbar.set_label('Number of Bursts', size=16)
        #cbar.set_ticks(np.array([1,2,3.5,5]) + 0.5)
        #cbar.set_ticklabels(np.array([1,2,3,5]))
        ax.plot(all_r, density_sg, color='black')
        ax.set_ylim((1000, density_sg.max()*5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.axvline(x=.038, linestyle='--', color='black', label='softening length')
        ax.axvline(x=2.8*.038, linestyle='--', color='gray')
        ax.set_xlabel('Radius (kpc)')
        ax.text(0.08, 1e4, r'\textbf{2.8 $\times$ softening}', rotation='vertical', color='gray', fontsize=15)
        ax.legend(frameon=False, title=r'\textbf{Number of Bursts}', title_fontsize=15, loc='upper right')
        ax.set_ylabel('Density ($M_{\odot}$/kpc$^3$)')
        ax.text(0.98, 0.05, r'Fixed total outflow mass = $1.8 \times 10^{8}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, fontsize=15)
        ax.text(0.98, 0.12, r'$M_{\rm halo} = 9.2 \times 10^{9}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=15)
        ax.text(0.98, 0.19, r'$M_{*} = 3.9 \times 10^{6}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=15)
        fig.savefig('density.pdf', bbox_inches='tight')
        return


if __name__=='__main__':
	main()
 
