import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from copy import copy

from readData.DataLoader import DataLoader 
from analysis import analyze

import visualization.contour_makepic as cmakepic
import util.calc_hsml as calc_hsml

h = 0.6909
time = 1
boxsize = np.array([25000,25000,25000])

def wrap(center, coords, Lbox):
    dx = coords[:,0] - center[0]
    dy = coords[:,1] - center[1]
    dz = coords[:,2] - center[2]

    # x-coordinate
    mask = (dx > Lbox[0]/2.0)
    dx[mask] = dx[mask] - Lbox[0]
    mask = (dx < -Lbox[0]/2.0)
    dx[mask] = dx[mask] + Lbox[0]

    # y-coordinate
    mask = (dy > Lbox[1]/2.0)
    dy[mask] = dy[mask] - Lbox[1]
    mask = (dy < -Lbox[1]/2.0)
    dy[mask] = dy[mask] + Lbox[1]

    # z-coordinate
    mask = (dz > Lbox[2]/2.0)
    dz[mask] = dz[mask] - Lbox[2]
    mask = (dz < -Lbox[2]/2.0)
    dz[mask] = dz[mask] + Lbox[2]

    # format coordinates
    coords = np.vstack((dx,dy,dz)).T

    return coords

def get_massmap(cat, pixels, fov=50, face_on=False, edge_on=False, part_types=[4], filt=["V"]):

    coords = copy(cat['Coordinates']) / h * time
    vels = copy(cat['Velocities']) * np.sqrt(time)
    masses = cat['Masses'] * 1e10 / h

    gal_pos = cat['GroupPos'] / h * time
    gal_vel = cat['GroupVel'] * np.sqrt(time)

    coords = wrap(gal_pos, coords, boxsize/h*time)
    vels -= gal_vel

    hsml = None

    if type(fov) == type([]):
        xrange=[-fov[0],fov[0]]
        yrange=[-fov[1],fov[1]]
        maxfov = np.max(fov)
    else:
        xrange=[-fov,fov]
        yrange=[-fov,fov]
        maxfov = fov

    weights = masses

    cut = analyze.get_box_cut(coords, np.zeros(3), maxfov)

    if hsml is None:
        if 'SubfindHsml' in cat:
            hsml = cat['SubfindHsml'][cut]
        else:
            hsml = calc_hsml.get_particle_hsml(coords[cut,0], coords[cut,1], coords[cut,2], DesNgb=32, Hmax=50)

    massmap,image = cmakepic.simple_makepic(coords[cut,0], coords[cut,1],
            weights=weights[cut], hsml=hsml,xrange=xrange,yrange=yrange, pixels=pixels)

    print(massmap.shape)
    return massmap



def makeimg(path):

    #name = f'm10dmohalo'

    just_halo = False

    #path to simulation output
    #path = f"/home/oliviamostow/Projects/b31_s025_hr/s7/mb_mass10-output"

    part_types = [1,2]
    keys = ["Coordinates","GroupPos","GroupVel", "Velocities", "Masses", 'SubfindHsml'] 

    pixels = 512

    #choose which halo to view
    sub_idx = 0
    fof_idx = 0

    #choose which snapshot
    snap_num = 127

    #load in the data
    if just_halo: 
        cat = DataLoader(path, part_types=part_types, snap_num=snap_num, keys=keys, sub_idx=sub_idx, fof_idx=fof_idx) 
    else:
        keys = ['Coordinates', 'Masses', 'Velocities','GFM_StellarPhotometrics', 'SubfindHsml', 'ParticleIDs']

        pcat = DataLoader(path, part_types=part_types, snap_num=snap_num, keys=keys) 
        gal_cat = DataLoader(path, part_types=part_types, snap_num=snap_num, keys=['GroupPos', 'GroupVel'], fof_idx=fof_idx, sub_idx=sub_idx)            

        #combine low-res and high-res particle types
        if 1 in part_types and 2 in part_types:
            part_cat = dict()
            for key in keys:
                if key not in pcat:
                    continue
                part_cat[key] = np.concatenate([pcat[f'PartType1/{key}'], pcat[f'PartType2/{key}']], axis=0)
            pcat = part_cat


        cat = {'Coordinates': pcat['Coordinates'],
                'Velocities': pcat['Velocities'],
                'Masses': pcat['Masses'],
                'GroupPos': gal_cat['GroupPos'],
                'GroupVel': gal_cat['GroupVel']}


        if 'SubfindHsml' in pcat:
            cat['SubfindHsml'] = pcat['SubfindHsml'][box_cut]


    #define the field of view in kpc
    if just_halo:
        fov = 15
    else:
        fov = 15

    #make the massmap
    massmap = get_massmap(cat, pixels=pixels, fov=fov, part_types=part_types)

    #make it pretty for plotting
    massmap[np.isnan(massmap)] = 1e0
    massmap[massmap<=1e0] = 1e0

    cdict = {
    'red'  :  ((0., 0., 0.),     (0.3,0,0),     (0.6, 0.8, 0.8), (1., 1., 1.)),
    'green':  ((0., 0., 0.),     (0.3,0.3,0.3), (0.6, 0.4, 0.4), (1., 1.0, 1.0)),
    'blue' :  ((0., 0.05, 0.05), (0.3,0.5,0.5), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0))
    }
    dmdens_cmap = LinearSegmentedColormap('dmdens_cmap', cdict, 1024)

    #plot the image
    #fig, ax = plt.subplots(figsize=(5,5))
    print(massmap.shape)
    #im = ax.imshow(massmap, norm=LogNorm(vmin=1e2, vmax=6e8), cmap=dmdens_cmap)
    #ax.set_xticks([])
    #ax.set_yticks([])           

    #save the image
    #fig.savefig(f"plots/uniform_map_{name}.pdf", bbox_inches='tight')


    return massmap, dmdens_cmap

def main():
    fig, ax = plt.subplots(ncols=2)
    name = f'm10multi'
    paths_list = ['../m10_ic0_sg_713/output','../fsm_b31_713_s025/s8/output']
    for i in range(len(paths_list)):
        massmap, dmdens_cmap = makeimg(paths_list[i])
        im = ax[i].imshow(massmap, norm=LogNorm(vmin=1e6, vmax=6e8), cmap=dmdens_cmap)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.subplots_adjust(wspace=.1)
    fig.savefig(f"plots/uniform_map_{name}.pdf", bbox_inches='tight')
    return

if __name__=="__main__":
    main()

