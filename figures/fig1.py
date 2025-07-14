
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from  astropy.cosmology import FlatLambdaCDM
import pandas as pd
import shared_data
import sg
import makehaloimg as himg
import calculations
from matplotlib.colors import LogNorm,LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.gridspec as gridspec


#for density plots
rmin = 0.038
rmax = 10
snap = 127
sphere_samples = 200
radial_samples = 50
DesNgb = 32
h = 0.6909

def main():
        plt.rc('font',**{'family':'STIXGeneral'})
        plt.rc('text', usetex=True)

        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
        plt.rc('legend', frameon=False, fontsize=16)
        plt.rcParams["figure.frameon"] = False
        plt.rc('lines', linewidth=3)
        fig = plt.figure(figsize=(20,20))
        gs = gridspec.GridSpec(5,3, height_ratios=[1,1,0.2,1,1], width_ratios=[0.5,0.5,1])
        ax_cusp_img  = fig.add_subplot(gs[0:2,0:2])
        ax_cusp_mass = fig.add_subplot(gs[0,2])
        ax_cusp_dens = fig.add_subplot(gs[1,2])
        ax_core_img  = fig.add_subplot(gs[3:5,0:2])
        ax_core_mass = fig.add_subplot(gs[3,2])
        ax_core_dens = fig.add_subplot(gs[4,2])
        
        #cosmology
        cosmo = FlatLambdaCDM(H0=69.09, Om0=.3, Tcmb0=2.725)
        minsnap=20
        #tracer mass plots
        #read in needed data
        df1 = pd.read_csv('shmr_t.txt', header=None)
        df2 = pd.read_csv('tracer_t.txt', header=None)
        stellar_mass = df1[1]
        t = df1[0]

        ax_cusp_mass.set_yscale('log')
        ax_cusp_mass.set_ylim(1e3,5e6)
        ax_cusp_mass.invert_xaxis()
        ax_cusp_mass.set_ylabel('Mass ($M_{\odot}$)')
        ax_cusp_mass.set_xlabel('Lookback Time (Gyr)')
        ts, stellar_t = sg.get_points(1000, sg.smooth2)
        ax_cusp_mass.plot(cosmo.lookback_time(t[minsnap:]), stellar_mass[minsnap:], color='black', label='$M_{*}$ from SHMR')
        ax_cusp_mass.plot(cosmo.lookback_time(1/ts - 1), stellar_t, color='orchid', label='Tracer mass')
        ax_cusp_mass.legend(frameon=False)
        
        ax_core_mass.set_yscale('log')
        ax_core_mass.invert_xaxis()
        ax_core_mass.set_ylabel('Mass ($M_{\odot}$)')
        ax_core_mass.set_xlabel('Lookback Time (Gyr)')
        ts, stellar_t = sg.get_points(2000, sg.b5)
        ax_core_mass.plot(cosmo.lookback_time((1/ts) - 1), stellar_t, color='blue', label='Tracer mass')
        ax_core_mass.legend()
        
        #adding a redshift axis to both plots
        ax2_core_mass = ax_core_mass.twiny()
        ax2_cusp_mass = ax_cusp_mass.twiny()
        newlabel = [12,6,4,3,2,1]
        convert = lambda x : cosmo.lookback_time(x).value
        newpos = [convert(z) for z in newlabel]
        ax2_core_mass.set_xticks(newpos)
        ax2_core_mass.set_xticklabels(newlabel)
        ax2_core_mass.invert_xaxis()
        ax2_core_mass.set_xlabel('Redshift')
        ax2_core_mass.set_xlim(ax_core_mass.get_xlim())
        ax2_cusp_mass.set_xticks(newpos)
        ax2_cusp_mass.set_xticklabels(newlabel)
        ax2_cusp_mass.invert_xaxis()
        ax2_cusp_mass.set_xlabel('Redshift')
        ax2_cusp_mass.set_xlim(ax_cusp_mass.get_xlim())

        #density plots
        h = 0.6909
        all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
        sgpath = '../m10_ic0_sg_713/output'
        sg_density = []
        sgcat = DataLoader(sgpath, snap, 1, ['Masses','Coordinates','SubhaloPos'], sub_idx=0)
        sgcoords = sgcat['PartType1/Coordinates']/h - sgcat['SubhaloPos']/h
        sgmasses = sgcat['PartType1/Masses']*1e10/h
        for r in all_r:
            points = calculations.fibonacci_sphere(sphere_samples, r)
            density = calculations.calc_density(sgcoords, sgmasses, points, DesNgb)
            sg_density.append(density)
        sg_density = np.array(sg_density)
        ax_cusp_dens.plot(all_r, sg_density, color='orchid')
        ax_cusp_dens.set_yscale('log')
        ax_cusp_dens.set_xscale('log')
        ax_cusp_dens.axvline(x=0.038*2.8, linestyle='dashed', color='grey')
        ax_cusp_dens.legend(frameon=False)
        ax_cusp_dens.set_ylabel('Density ($M_{\odot}$/kpc$^3$)')
        ax_cusp_dens.set_ylim(1e4, 1e9)
        ax_cusp_dens.set_xlabel('Radius (kpc)')
        bpath = '../updated_m10_713s/s8b5/output'
        b_density = []
        bcat = DataLoader(bpath, snap, 1, ['Masses','Coordinates','SubhaloMass','SubhaloPos'], sub_idx=0)
        bcoords = bcat['PartType1/Coordinates']/h - bcat['SubhaloPos']/h
        bmasses = bcat['PartType1/Masses']*1e10/h
        for r in all_r:
            points = calculations.fibonacci_sphere(sphere_samples, r)
            density = calculations.calc_density(bcoords, bmasses, points, DesNgb)
            b_density.append(density)
        b_density = np.array(b_density)
        ax_core_dens.plot(all_r, b_density, color='blue')
        ax_core_dens.axvline(x=0.038*2.8, linestyle='dashed', color='grey')
        ax_core_dens.set_xlabel('Radius (kpc)')
        ax_core_dens.set_ylabel('Density ($M_{\odot}$/kpc$^3$)')
        ax_core_dens.legend(frameon=False)
        ax_core_dens.set_yscale('log')
        ax_core_dens.set_xscale('log')
        ax_core_dens.set_ylim(1e4,1e9)

        ax_core_dens.text(0.08, 5e4, r'\textbf{2.8 $\times$ softening}', rotation='vertical', color='gray', fontsize=15)
        ax_cusp_dens.text(0.08, 5e4, r'\textbf{2.8 $\times$ softening}', rotation='vertical', color='gray', fontsize=15)

        #halo images
        massmap1, dmdens1 = himg.makeimg(sgpath)
        im = ax_cusp_img.imshow(massmap1, norm=LogNorm(vmin=1e5, vmax=6e8), cmap=dmdens1)
        ax_cusp_img.set_xticks([])
        ax_cusp_img.set_yticks([])
        ax_cusp_img.plot(np.arange(50,170 + 50), np.repeat(440, 170), 30, color='white')
        massmap2, dmdens2 = himg.makeimg(bpath)
        im = ax_core_img.imshow(massmap2, norm=LogNorm(vmin=1e5, vmax=6e8), cmap=dmdens2)
        ax_core_img.set_xticks([])
        ax_core_img.set_yticks([])
        ax_core_img.plot(np.arange(50,170 + 50), np.repeat(440, 170), 30, color='white')
        ax_cusp_img.text(90,490, '10 kpc', color='white', size=32)
        ax_core_img.text(90,490,'10 kpc', color='white', size=32)
        ax_cusp_img.text(0.6, 0.90, r'{$M_{\rm halo} = 9.2 \times 10^{9}~M_{\odot}$}',verticalalignment='bottom', horizontalalignment='right',transform=ax_cusp_img.transAxes, fontsize=28, color='white')
        ax_core_img.text(0.6, 0.90, r'{$M_{\rm halo} = 9.2 \times 10^{9}~M_{\odot}$}',verticalalignment='bottom', horizontalalignment='right',transform=ax_core_img.transAxes, fontsize=28, color='white')
        ax_cusp_img.text(0.55, 0.80, r'{$M_{*} = 3.9 \times 10^{6}~M_{\odot}$}',verticalalignment='bottom', horizontalalignment='right',transform=ax_cusp_img.transAxes, fontsize=28, color='white')
        ax_core_img.text(0.55, 0.80, r'{$M_{*} = 3.9 \times 10^{6}~M_{\odot}$}',verticalalignment='bottom', horizontalalignment='right',transform=ax_core_img.transAxes, fontsize=28, color='white')
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        #plt.subplots(layout='constrained')
        ax_core_img.set_title('Bursty Model', loc='center', size=36)
        ax_cusp_img.set_title('Smooth Model', loc='center', size=36)
        fig.savefig('f1up.pdf', bbox_inches='tight') 
        return


if __name__=='__main__':
	main()

