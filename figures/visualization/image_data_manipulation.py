import simread.readsubfHDF5 as readsubf
import numpy as np


def center_data( data, **kwargs):
    # if "center" is set, this overrules everything else.
    if (kwargs.get("center") is not None) and (kwargs.get("center") is not [0,0,0]):
        for ijk in range(3):  data.pos[:,ijk] -= center[ijk]
        return data

    if (kwargs.get("cosmo") is not None) and (kwargs.get("cosmo")>0):
        if ((kwargs.get("fof_num") is None) and (kwargs.get("sub_num") is None) ): # neither set.  Center on middle of full box.
            boxsize = data.header.boxsize
            for ijk in range(3):  data.pos[:,ijk] -= boxsize/2.0
            return data
        else:   # we are loading a specific fof and/or subhalo
            if (kwargs.get("fof_num") is not None): fof_num = kwargs.get("fof_num")
            else: fof_num=-1
            if (kwargs.get("sub_num") is not None): sub_num = kwargs.get("sub_num")
            else: sub_num=-1

            subfind_cat = readsubf.subfind_catalog(  data.snapdir, data.snapnum, subcat=True, grpcat=True,
                       keysel=['SubhaloPos', 'GroupFirstSub'] )

            if(fof_num >= 0):
                GroupFirstSub = subfind_cat.GroupFirstSub[fof_num]
                if (sub_num >= 0):
                    sub_num += GroupFirstSub
                else:
                    sub_num = GroupFirstSub

            center = subfind_cat.SubhaloPos[sub_num, :]
            for ijk in range(3):  data.pos[:,ijk] -= center[ijk]

    return data


def clip_particles( data, xr, yr, zr, **kwargs ):
    print("Clipping paticles outside of range xr=[{:8.4f},{:8.4f}] and yr=[{:8.4f},{:8.4f}] and zr=[{:8.4f},{:8.4f}]".format( xr[0], xr[1], yr[0], yr[1], zr[0], zr[1] ) )
    print("Prior to clipping we have {:d} particles".format( len( data.pos[:,0] ) )  )

    okay_index = ( data.pos[:,0] > xr[0] ) & \
                 ( data.pos[:,0] < xr[1] ) & \
                 ( data.pos[:,1] > yr[0] ) & \
                 ( data.pos[:,1] < yr[1] ) & \
                 ( data.pos[:,2] > zr[0] ) & \
                 ( data.pos[:,2] < zr[1] )


    for field in dir(data):
        if type(getattr(data,field)) is np.ndarray:
            tmp = getattr(data,field)
            setattr( data, field, tmp[okay_index] )

    print("After clipping we have {:d} particles".format( len( data.pos[:,0] ) )  )

    return data


def rotate_data( data, phi=0, theta=0, **kwargs):
        x = data.pos[:,0]
        y = data.pos[:,1]
        z = data.pos[:,2]
        vx = data.vel[:,0]
        vy = data.vel[:,1]
        vz = data.vel[:,2]

        x_ = -z  * np.sin(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.cos(theta)
        y_ = -x  * np.sin(phi)   + y  * np.cos(phi)
        z_ =  z  * np.cos(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.sin(theta)
        vx_ = -vz  * np.sin(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.cos(theta)
        vy_ = -vx  * np.sin(phi)   + vy  * np.cos(phi)
        vz_ =  vz  * np.cos(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.sin(theta)

        data.pos[:,0] = x_
        data.pos[:,1] = y_
        data.pos[:,2] = z_

        data.vel[:,0] = vx_
        data.vel[:,1] = vy_
        data.vel[:,2] = vz_

        return data

def rescale_hsml( data, h_rescale_factor=1.0, **kwargs):
    data.h = data.h * h_rescale_factor
    return data
