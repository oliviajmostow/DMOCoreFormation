
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from matplotlib.colors import LogNorm
import shared_data
import calculations
import cd_function

rmin = 0.038
rmax = 30
snap = 127
sphere_samples = 200
radial_samples = 30
DesNgb = 32
h = 0.6909
path = '../m8_sg_g2_038/output'
all_density = []
cat = DataLoader(path, snap, 1, ['Masses','Coordinates','GroupPos','SubhaloMass','SubhaloPos', 'GroupMassType'], sub_idx=0)
coords = cat['PartType1/Coordinates']/h - cat['SubhaloPos']/h
masses = cat['PartType1/Masses']*1e10/h
def main():
        all_density = []
        #smooth
        all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
        for r in all_r:
            points = calculations.fibonacci_sphere(sphere_samples, r)
            density = calculations.calc_density(coords, masses, points, DesNgb)
            all_density.append(density)
        all_density = np.array(all_density)      
        #plotting
        fig, ax = shared_data.set_plot_params()
        ax.plot(all_r, all_density, color='black', label='Smooth model')
        
        #single burst
        path = '../m8soft038/m1_r7/sb_mass8-output'
        all_density = []
        cat2 = DataLoader(path, snap, 1, ['Masses','Coordinates','GroupPos','SubhaloMass','SubhaloPos', 'GroupMassType'], sub_idx=0)
        coords2 = cat2['PartType1/Coordinates']/h - cat2['SubhaloPos']/h
        masses2 = cat2['PartType1/Masses']*1e10/h
        for r in all_r:
            points = calculations.fibonacci_sphere(sphere_samples, r)
            density = calculations.calc_density(coords2, masses2, points, DesNgb)
            all_density.append(density)
        all_density = np.array(all_density)
        #ax.plot(all_r, all_density, color='red', label='Single-burst')
        ax.set_ylim((1000, all_density.max()*3))
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.axvline(x=.038, linestyle='--', color='black', label='softening length')
        ax.axvline(x=2.8*.038, linestyle='--', color='gray')
        ax.text(0.08, 1e4, r'\textbf{2.8 $\times$ softening}', rotation='vertical',fontsize=15, color='gray')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Density ($M_{\odot}$/kpc$^3$)')
        ax.legend(frameon=False)
        #ax.text(0.98, 0.05, r'Ultra-faint Dwarf',verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, fontsize=15)
        ax.text(0.95, 0.52, r'$M_{\rm halo} = 7.8 \times 10^{7}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=15)
        ax.text(0.95, 0.59, r'$M_{*} = 1.4 \times 10^{3}~M_{\odot}$',verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes, fontsize=15)
        fig.savefig('density.pdf', bbox_inches='tight')
        return


if __name__=='__main__':
	main()

