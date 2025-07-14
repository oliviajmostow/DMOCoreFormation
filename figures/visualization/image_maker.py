import numpy as np
import matplotlib
matplotlib.use('agg') ## this calls matplotlib without a X-windows GUI

from subprocess import call
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import math
import util.utilities as util
import gadget_lib.gadget as gadget
import gadget_lib.load_stellar_hsml as starhsml
import visualization.colors as viscolors
import visualization.get_attenuated_stellar_luminosities as getlum
import visualization.make_threeband_image as makethreepic
import visualization.contour_makepic as cmakepic
import visualization.raytrace_projection as rayproj
import h5py

import util.calc_hsml as calc_hsml
import simread.readsubfHDF5 as readsubf

import simread.readsnapHDF5 as rs
import units.springel_units as units

import util.naming as naming

import visualization.image_data_manipulation as idm
import visualization.image_properties as ip

def checklen(x):
    return len(np.array(x,ndmin=1));

def ok_scan(input,xmax=1.0e10,xmin=-1.0e10,pos=0):
    if (pos==1):
        return (np.isnan(input)==False) & (np.abs(input)<=xmax) & (input>=xmin) & (input > 0.);
    else:
        return (np.isnan(input)==False) & (np.abs(input)<=xmax) & (np.abs(input)>=xmin);

def snap_ext(snum,four_char=0):
	ext='00'+str(snum);
	if (snum>=10): ext='0'+str(snum)
	if (snum>=100): ext=str(snum)
	if (four_char==1): ext='0'+ext
	if (snum>=1000): ext=str(snum)
	return ext;

def get_gas_xray_luminosity(ppp):
    brems = gadget.gas_xray_brems( \
        ppp['m'], ppp['u'], ppp['rho'], ppp['ne'], ppp['nh'] );
    return brems; ## can also add metal line cooling luminosities later



def load_illustris_subbox_brick( snapdir, snapnum, subbox, load_all_elements=False):
    filename = snapdir + 'snapdir_subbox'+str(subbox)+'_'+str(snapnum).zfill(3)+'/snap_subbox'+str(subbox)+'_'+str(snapnum).zfill(3)+'.0.hdf5'

    with h5py.File(filename) as f:
        header = dict( f['Header'].attrs.items() )
        npart_total = header['NumPart_Total']
        num_files   = header['NumFilesPerSnapshot']

    u_all = np.zeros( npart_total[0] )
    rho_all = np.zeros( npart_total[0] )
    m_all = np.zeros( npart_total[0] )
    h_all = np.zeros( npart_total[0] )
    nh_all = np.zeros( npart_total[0] )
    ne_all = np.zeros( npart_total[0] )
    sfr_all = np.zeros( npart_total[0] )
    lx_all = np.zeros( npart_total[0] )
    zm_all = np.zeros( npart_total[0] )
    c_all = np.zeros( npart_total[0] )

    if load_all_elements:
        zm9_all = np.zeros( (npart_total[0], 9) )

    pos_all = np.zeros( (npart_total[0], 3 ) )
    vel_all = np.zeros( (npart_total[0], 3 ) )

    index = 0
    for iii in range( num_files ):
        this_file = snapdir + 'snapdir_subbox'+str(subbox)+'_'+str(snapnum).zfill(3)+'/snap_subbox'+str(subbox)+'_'+str(snapnum).zfill(3)+'.'+str(iii)+'.hdf5'
        with h5py.File( this_file ) as f:
            header = dict( f['Header'].attrs.items() )

            if header['NumPart_ThisFile'][0] > 0:
                u_all[  index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/InternalEnergy'][:]
                rho_all[index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/Density'][:]
                m_all[  index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/Masses'][:]
                nh_all[ index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/NeutralHydrogenAbundance'][:]
                ne_all[ index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/ElectronAbundance'][:]
                sfr_all[index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/StarFormationRate'][:]
                zm_all[ index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/GFM_Metallicity'][:]
                c_all[  index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/InternalEnergy'][:]
                if load_all_elements:
                    zm9_all[index : index + header['NumPart'][0] ] = f['PartType0/InternalEnergy'][:, :9]
                pos_all[index : index + header['NumPart_ThisFile'][0], : ] = f['PartType0/Coordinates'][:, :]
                vel_all[index : index + header['NumPart_ThisFile'][0], : ] = f['PartType0/Velocities'][:, :]
                index += header['NumPart_ThisFile'][0]
            time = header['Time']

    h_all   = m_all / rho_all
    h_all   = 2.0 * 4.0* h_all ** 0.333 / 5.0
#    lx_all[ index : index + header['NumPart_ThisFile'][0] ] = f['PartType0/InternalEnergy'][:]

    x_all   = pos_all[:,0]
    y_all   = pos_all[:,1]
    z_all   = pos_all[:,2]
    vx_all   = vel_all[:,0]
    vy_all   = vel_all[:,1]
    vz_all   = vel_all[:,2]

    if load_all_elements:
        return     np.array(u_all,dtype='f'),   np.array(rho_all,dtype='f'), np.array(h_all,dtype='f'),  np.array(nh_all,dtype='f'), \
                   np.array(ne_all,dtype='f'), \
                   np.array(sfr_all,dtype='f'), np.array(lx_all,dtype='f'),  np.array(zm_all,dtype='f'), np.array(m_all,dtype='f'), \
                   np.array(x_all,dtype='f'),   np.array(y_all,dtype='f'),   np.array(z_all,dtype='f'),  \
                   np.array(vx_all,dtype='f'),   np.array(vy_all,dtype='f'),   np.array(vz_all,dtype='f'),  \
                   np.array(c_all,dtype='f'), np.array(zm9_all), \
                   time

    else:
        return     np.array(u_all,dtype='f'),   np.array(rho_all,dtype='f'), np.array(h_all,dtype='f'),  np.array(nh_all,dtype='f'), \
                   np.array(ne_all,dtype='f'), \
                   np.array(sfr_all,dtype='f'), np.array(lx_all,dtype='f'),  np.array(zm_all,dtype='f'), np.array(m_all,dtype='f'), \
                   np.array(x_all,dtype='f'),   np.array(y_all,dtype='f'),   np.array(z_all,dtype='f'),  \
                   np.array(vx_all,dtype='f'),   np.array(vy_all,dtype='f'),   np.array(vz_all,dtype='f'),  \
                   np.array(c_all,dtype='f'), \
                   time


def load_illustris_brick( snapdir, snapnum, fof_num, sub_num, load_all_elements=False, do_stars=False, center=False):
    import simread.readsnapHDF5 as ws
    import simread.readhaloHDF5 as hr
    import util.naming as snap_names

    try:
        run = snapdir[snapdir.index('Runs/')+5:snapdir.index('output')-1]
    except:
        run=None

    halo_reader = hr.HaloReader( snapdir, snapnum, run=run )
    
    if center:
        image_center = readsubf.subfind_catalog(  snapdir, snapnum, subcat=True, grpcat=True,
                       keysel=['SubhaloPos', 'GroupFirstSub'] )
        if(sub_num >= 0):
            sub_num_for_center = sub_num
        else:
            sub_num_for_center = 0

        if(fof_num >= 0):
            GroupFirstSub = image_center.GroupFirstSub[fof_num]
            sub_num_for_center += GroupFirstSub
        print("Loading center for Subhalo {:d} Position".format( sub_num_for_center ) ) 

        center = image_center.SubhaloPos[sub_num_for_center, :]
    else:
        center = [0,0,0]
    try:
        filename = snapdir + 'snapdir_'+str(snapnum).zfill(3)+'/snap_'+str(snapnum).zfill(3)
        time    = ws.snapshot_header(filename+'.0.hdf5')
        time = time.time
    except:
        filename = snapdir + 'snapdir_'+str(snapnum).zfill(3)+'/snapshot_'+str(snapnum).zfill(3)
        time    = ws.snapshot_header(filename )
        time = time.time

    if do_stars:
        m_all   = halo_reader.read( 'MASS', 4, fof_num, sub_num )
        pos_all = halo_reader.read( 'POS ', 4, fof_num, sub_num )
        vel_all = halo_reader.read( 'VEL ', 4, fof_num, sub_num )

        h=calc_hsml.get_particle_hsml(
                pos_all[:,0], pos_all[:,1], pos_all[:,2] )

        x_all   = pos_all[:,0] - center[0]
        y_all   = pos_all[:,1] - center[1]
        z_all   = pos_all[:,2] - center[2]
        vx_all   = vel_all[:,0]
        vy_all   = vel_all[:,1]
        vz_all   = vel_all[:,2]

        import util.cosmo_tools as ct

        try:    form_time_list = halo_reader.read('GAGE', 4, fof_num, sub_num)
        except: form_time_list = halo_reader.read('AGE ', 4, fof_num, sub_num)

        try:    zm_main = halo_reader.read('GZ  ', 4, fof_num, sub_num)
        except: zm_main = halo_reader.read('Z   ', 4, fof_num, sub_num)

        z_form = 1.0/form_time_list - 1.0
        z_now  = 1.0/time - 1.0

        t_form = ct.quick_lookback_time( z_form )
        t_now  = ct.quick_lookback_time( z_now  )
        c_main = t_now - t_form     # age of stellar population in Gyrs.

        c_main[ np.isnan( c_main ) ] = -1
        m_all[ c_main <= 0 ] = 0

        return  np.array(m_all,dtype='f'), np.array(h, dtype='f'),  \
                       np.array(x_all,dtype='f'),   np.array(y_all,dtype='f'),   np.array(z_all,dtype='f'),  \
                       np.array(vx_all,dtype='f'),   np.array(vy_all,dtype='f'),   np.array(vz_all,dtype='f'),  \
                       np.array(c_main,dtype='f'), np.array(zm_main,dtype='f'), \
                       time
    else:
        u_all   = halo_reader.read( 'U   ', 0, fof_num, sub_num )
        rho_all = halo_reader.read( 'RHO ', 0, fof_num, sub_num )
        m_all   = halo_reader.read( 'MASS', 0, fof_num, sub_num )

        h_all   = m_all / rho_all
        h_all   = 2.0 * 4.0* h_all ** 0.333 / 5.0
        nh_all  = halo_reader.read( 'NH  ', 0, fof_num, sub_num )
        ne_all  = halo_reader.read( 'NE  ', 0, fof_num, sub_num )
        sfr_all = halo_reader.read( 'SFR ', 0, fof_num, sub_num )
        lx_all  = np.zeros_like(u_all)					# lray luminosity
        zm_all  = halo_reader.read( 'GZ  ', 0, fof_num, sub_num )
        pos_all = halo_reader.read( 'POS ', 0, fof_num, sub_num )
        vel_all = halo_reader.read( 'VEL ', 0, fof_num, sub_num )
        c_all   = halo_reader.read( 'RHO ', 0, fof_num, sub_num )
        if load_all_elements:
            zm9_all = halo_reader.read('GMET', 0, fof_num, sub_num )


        x_all   = pos_all[:,0] - center[0]
        y_all   = pos_all[:,1] - center[1]
        z_all   = pos_all[:,2] - center[2]
        vx_all   = vel_all[:,0]
        vy_all   = vel_all[:,1]
        vz_all   = vel_all[:,2]

        if load_all_elements:
            return     np.array(u_all,dtype='f'),   np.array(rho_all,dtype='f'), np.array(h_all,dtype='f'),  np.array(nh_all,dtype='f'), \
                       np.array(ne_all,dtype='f'), \
                       np.array(sfr_all,dtype='f'), np.array(lx_all,dtype='f'),  np.array(zm_all,dtype='f'), np.array(m_all,dtype='f'), \
                       np.array(x_all,dtype='f'),   np.array(y_all,dtype='f'),   np.array(z_all,dtype='f'),  \
                       np.array(vx_all,dtype='f'),   np.array(vy_all,dtype='f'),   np.array(vz_all,dtype='f'),  \
                       np.array(c_all,dtype='f'), np.array(zm9_all), \
                       time

        else:
            return     np.array(u_all,dtype='f'),   np.array(rho_all,dtype='f'), np.array(h_all,dtype='f'),  np.array(nh_all,dtype='f'), \
                       np.array(ne_all,dtype='f'), \
                       np.array(sfr_all,dtype='f'), np.array(lx_all,dtype='f'),  np.array(zm_all,dtype='f'), np.array(m_all,dtype='f'), \
                       np.array(x_all,dtype='f'),   np.array(y_all,dtype='f'),   np.array(z_all,dtype='f'),  \
                       np.array(vx_all,dtype='f'),   np.array(vy_all,dtype='f'),   np.array(vz_all,dtype='f'),  \
                       np.array(c_all,dtype='f'), \
                       time



def center_position( x_all, y_all, z_all, center_on_bh=0, center_on_cm=0, center_pos=None, **kwargs ):
    print("WARNING!!! CENTER_POSITIONS IS ONLY ABLE TO HANDLE A center_pos ARGUMENT!")

    if center_on_bh>0:
        bh_pos = rs.read_block( snapshot, 'POS ', 5)
        bh_id  = rs.read_block( snapshot, 'POS ', 5)
        if center_on_bh==1:
            center_pos = np.array(bh_pos[np.where( bh_id == np.min(bh_id))[0],:]).flatten()
        elif center_on_bh==2:
            center_pos = np.array(bh_pos[np.where( bh_id == np.max(bh_id))[0],:]).flatten()
        else:
            print("center_on_bh was set, but its not clear which BH you want to be centered on")
            sys.exit()

    if center_on_cm==1:
        center_pos=[np.median(x_all),np.median(y_all),np.median(z_all)]

    if center_pos is None:
        center_pos=[0., 0., 0.]

    x_all -= center_pos[0]
    y_all -= center_pos[1]
    z_all -= center_pos[2]

    return x_all, y_all, z_all



def overlay_scale_label(xr,yr,figure_axis,c0='w'):
    ddx=0.5*(xr[1]-xr[0]);
    if (ddx>0.00002): dx=0.00001; dxl='0.01 pc';
    if (ddx>0.0002): dx=0.0001; dxl='0.1 pc';
    if (ddx>0.002): dx=0.001; dxl='1 pc';
    if (ddx>0.02): dx=0.01; dxl='10 pc';
    if (ddx>0.2): dx=0.1; dxl='100 pc';
    if (ddx>2.): dx=1.; dxl='1 kpc';
    if (ddx>20.): dx=10.; dxl='10 kpc';
    if (ddx>200.): dx=100.; dxl='100 kpc';
    if (ddx>2000.): dx=1000.; dxl='1 Mpc';
    if (ddx>20000.): dx=10000.; dxl='10 Mpc';

    xlen=(xr[1]-xr[0])
    ylen=(yr[1]-yr[0])
    xoff = (0.25+0.02)*ddx / xlen
    yoff = 0.025*(yr[1]-yr[0]) / ylen
    xr_new = np.array([xoff-dx/xlen*0.5,xoff+dx/xlen*0.5])
    yr_new = np.array([yoff,yoff])

    plt.text(xoff,1.55*yoff,dxl,color=c0,\
        horizontalalignment='center',verticalalignment='baseline',\
        transform=figure_axis.transAxes,fontsize=32)
    figure_axis.autoscale(False)
    figure_axis.plot(xr_new,yr_new,color=c0,linewidth=4.0,\
        transform=figure_axis.transAxes)


def overlay_time_label(time, figure_axis, cosmo=0, c0='w', n_sig_add=0, tunit_suffix='Gyr'):
    if(cosmo==0):
        time_to_use = time ## absolute time in Gyr
        prefix = 't='
        suffix = 'Gyr'	#tunit
    else: ##
        time_to_use = 1./time - 1. ## redshift
        prefix = 'z='
        suffix = ''

    n_sig = 2
    if time_to_use >= 10: n_sig += 0
    if time_to_use < 1.: n_sig += 1
    t_str = round_to_n( time_to_use, n_sig+n_sig_add )
    label_str = prefix+t_str+suffix
    xoff=0.03; yoff=1.-0.025;
    import matplotlib.patheffects as path_effects

    plt.text(0.5,0.5,label_str,color=c0,\
        horizontalalignment='left',verticalalignment='top',\
        transform=figure_axis.transAxes,fontsize=32,
        path_effects=[path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])




def round_to_n(x, n):
    ''' Utility function used to round labels to significant figures for display purposes  from: http://mail.python.org/pipermail/tutor/2004-July/030324.html'''
    if n < 1:
        raise ValueError("number of significant digits must be >= 1")

    # show everything as floats (preference; can switch using code below to showing eN instead
    format = "%." +str(n-1) +"f"
    as_string=format % x
    return as_string

    # Use %e format to get the n most significant digits, as a string.
    format = "%." + str(n-1) + "e"
    as_string = format % x
    if as_string[-3:] in ['+00', '+01', '+02', '+03','-01', '-02', '-03']:
        #then number is 'small', show this as a float
        format = "%." +str(n-1) +"f"
        as_string=format % x
    return as_string


def coordinates_rotate(x_all, y_all, z_all, theta, phi, coordinates_cylindrical=0):
    ## set viewing angle for image
    ## project into plane defined by vectors perpendicular to the vector of this angle
    x = np.cos(phi)*x_all + np.sin(phi)*y_all + 0.*z_all
    y = -np.cos(theta)*np.sin(phi)*x_all + np.cos(theta)*np.cos(phi)*y_all + np.sin(theta)*z_all
    z =  np.sin(theta)*np.sin(phi)*x_all - np.sin(theta)*np.cos(phi)*y_all + np.cos(theta)*z_all

    if coordinates_cylindrical==1:
        x = x / np.abs(x)*np.sqrt(x*x+z*z)

    return x,y,z


def coordinates_transform_cylindrical( x, y, z ):
    x = np.append( x, -1.0*x)
    y = np.append( y, y)
    z = np.append( z, z)


#
# given a set of positions, project them to the coordinates they 'would be at'
#   with respect to a given camera (for ray-tracing purposes)
#
def coordinates_project_to_camera(x, y, z, \
        camera_pos=[0.,0.,0.], camera_dir=[0.,0.,1.], \
        screen_distance=1.0 ):
    camera_pos=np.array(camera_pos,dtype='d');
    camera_dir=np.array(camera_dir,dtype='d');
    ## recenter at camera location
    x-=camera_pos[0]; y-=camera_pos[1]; z-=camera_pos[2];
    ## line-of-sight distance along camera viewing angle
    znew = x*camera_dir[0] + y*camera_dir[1] + z*camera_dir[2];
    ## get perpendicular axes to determine position in 'x-y' space:
    cperp_x,cperp_y = util.return_perp_vectors(camera_dir);
    xnew = x*cperp_x[0] + y*cperp_x[1] + z*cperp_x[2];
    ynew = x*cperp_y[0] + y*cperp_y[1] + z*cperp_y[2];

    ## project to 'screen' at fixed distance along z-axis ::
    znew[znew==0.]=-1.0e-10; ## just to prevent nans
    ## need correction factors for other quantities (like hsml, for example)
    r_corr = screen_distance/znew;
    xnew *= r_corr; ## put positions 'at the screen'
    ynew *= r_corr; ## put positions 'at the screen'
    znew = -znew; ## so values in the direction of camera are *negative* (for later sorting)

    return xnew, ynew, znew, r_corr


def color2dimage( color, intensity, width=0.8, shift=0.05, rotation=2.0, csat = 1.0, n_rot = 0.666, flip_intensity=False):
        if flip_intensity:
            intensity = -1.0*intensity + 1.0

        this_lambda = intensity * width + (1. - width)/2.0 + shift
        this_phi    = color * 2. * 3.14159 * n_rot + 2.0 * 3.14159 * rotation / 3.0

        a = csat * this_lambda *(1.0-this_lambda)/2.0
        r = (this_lambda - a * 0.14861 * np.cos(this_phi)  + a * 1.78277 * np.sin(this_phi))
        g = (this_lambda - a * 0.29227 * np.cos(this_phi)  - a * 0.90649 * np.sin(this_phi))
        b = (this_lambda + a * 1.97294 * np.cos(this_phi)   )

        return r,g,b



def illustris_subbox_gas_image( snapdir, snapnum, subbox, pos,
                         xrange=0,yrange=0,zrange=0,
                         pixels=64,
                         include_lighting=1,
                         weighting='gas', face_on=False, edge_on=False,
                         little_h = 0.6774,
                         lx=None, ly=None, lz=None,
                         cmap=None,
                         **kwargs  ):

    # load and center data
    if weighting=='OxygenToNitrogen' or weighting=='IronToOxygen':
        u, rho, h, nh, ne, sfr, lxray, zm, m, x, y, z, vx, vy, vz, c, zm9, time = load_illustris_subbox_brick( snapdir, snapnum, subbox, load_all_elements=True )
    else:
        u, rho, h, nh, ne, sfr, lxray, zm, m, x, y, z, vx, vy, vz, c, time = load_illustris_subbox_brick( snapdir, snapnum, subbox )

    image_center = pos

    x -= pos[0]
    y -= pos[1]
    z -= pos[2]

    if (face_on==True) or (edge_on==True) or ((lx != None) and (ly !=  None) and (lz != None)):
        if ((lx != None) and (ly !=  None) and (lz != None)):
            print("ang mom vectors already set.  reusing.")
        else:
            r = np.sqrt( x**2 + y**2 + z**2 )
            index = (sfr > 0) & (r < 5.0)
            lz = np.sum( m[index] * (x[index] * vy[index] - y[index] * vx[index] ) )
            lx = np.sum( m[index] * (y[index] * vz[index] - z[index] * vy[index] ) )
            ly = np.sum( m[index] * (z[index] * vx[index] - x[index] * vz[index] ) )

        if face_on:
            phi   = np.arctan2( ly, lx )    #  + 3.14159/2.0
            theta =  np.arctan2( np.sqrt(lx**2 + ly**2), lz )      # + 3.14159/2.0
        if edge_on:
            phi   = np.arctan2( ly, lx ) + 3.14159/2.0
            theta = 3.14159/2.0 + np.arctan2( np.sqrt(lx**2 + ly**2), lz )	# + 3.14159/2.0


        x_ = -z  * np.sin(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.cos(theta)
        y_ = -x  * np.sin(phi)   + y  * np.cos(phi)
        z_ =  z  * np.cos(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.sin(theta)
        vx_ = -vz  * np.sin(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.cos(theta)
        vy_ = -vx  * np.sin(phi)   + vy  * np.cos(phi)
        vz_ =  vz  * np.cos(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.sin(theta)

        x = x_
        y = y_
        z = z_
        vx = vx_
        vy = vy_
        vz = vz_


    # trim particles to those inside/near the plotting region
    sidebuf=0.5;

    xr=xrange; yr=yrange; zr=zrange;
    if xr==0 and yr==0 and zr==0:
        xr=[np.min( x), np.max(x)]
        yr=[np.min( y), np.max(y)]
        zr=[np.min( z), np.max(z)]

    xlen=0.5*(xr[1]-xr[0]); ylen=0.5*(yr[1]-yr[0]); zlen=0.5*(zr[1]-zr[0])
    if (checklen(yrange)>1): yr=yrange;
    scale=0.5*np.max([xr[1]-xr[0],yr[1]-yr[0]]);
    if (checklen(zrange)>1): zr=zrange;
    else:  zr=np.array([-1.,1.])*scale;

    ok =    ok_scan(x,xmin=xr[0]-xlen*sidebuf, xmax=xr[1]+xlen*sidebuf) & \
            ok_scan(y,xmin=yr[0]-ylen*sidebuf, xmax=yr[1]+ylen*sidebuf) & \
            ok_scan(z,xmin=zr[0]-zlen*sidebuf, xmax=zr[1]+zlen*sidebuf) & \
            ok_scan(m,pos=1,xmax=1.0e40) & \
            ok_scan(h,pos=1) & ok_scan(c,pos=1,xmax=1.0e40)

    print("keeping {:d} particle out of {:d}".format( np.sum(ok), len(ok) ) )

#    np.min(np.fabs(xr))*sidebuf,xmax=np.max(np.fabs(xr))*sidebuf) & \


    if weighting=='gas':
            massmap,image_singledepth = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels)#,
               # **kwargs)
            massmap = massmap * 1e10 / 1e6 * little_h # solar masses  per cpc^2

            if ('set_maxden' in kwargs) and ('set_dynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxden'] ] = kwargs['set_maxden']
                massmap[ massmap < kwargs['set_maxden'] / kwargs['set_dynrng']  ] = kwargs['set_maxden'] / kwargs['set_dynrng']
                maxden = kwargs['set_maxden']
                dynrng = kwargs['set_dynrng']
                minden = maxden / dynrng
            else:
                minden = np.min( massmap )
                maxden = np.max( massmap )
                dynrng = maxden/minden

            print("before map adjustment, the massmap values are min/max/median = {:16.8f}/{:16.8f}/{:16.8f}".format( np.min( massmap), np.max(massmap), np.median(massmap) ) )
            gas_image = massmap
            gas_image = np.log10( gas_image )
            gas_image -= np.min( np.log10( minden ) )
            gas_image /= np.max( np.log10(maxden) - np.log10( minden ) )
            gas_image *= 253.
            gas_image += 1.0

            ## convert to actual image using a colortable:
            if cmap==None:  cmap='hot'
            my_cmap=matplotlib.cm.get_cmap(cmap);
            image24 = my_cmap(gas_image/255.);


            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cbar_fontsize = 40
            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(maxden/dynrng), np.log10(maxden), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(}\Sigma_\mathrm{gas}\mathrm{/[M_\odot\/pc^2])}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize)
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_gas_surface_density.pdf')


#            ## convert to actual image using a colortable:
#            image_singledepth[ image_singledepth < 2.0 ] = 255.0
#            my_cmap=matplotlib.cm.get_cmap('hot');
#            image24 = my_cmap(image_singledepth/255.);
            return image24, massmap, lx, ly, lz


    if weighting=='sfr':
            if np.sum(sfr[ok]) > 0.1:            # SFR of at least 0.1 solar masses per year
                massmap,image_singledepth = \
                cmakepic.simple_makepic(x[ok],y[ok],weights=sfr[ok],hsml=h[ok],\
                    xrange=xr,yrange=yr,
                    pixels=pixels)#,
                   # **kwargs)
                massmap = massmap * little_h * little_h # solar masses  per ckpc^2
            else:
                massmap = np.zeros( (pixels, pixels) )

            if ('set_maxsfrden' in kwargs) and ('set_sfrdynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxsfrden'] ] = kwargs['set_maxsfrden']
                massmap[ massmap < kwargs['set_maxsfrden'] / kwargs['set_sfrdynrng']  ] = kwargs['set_maxsfrden'] / kwargs['set_sfrdynrng']
                max_sfr = kwargs['set_maxsfrden']
                dynrng  = kwargs['set_sfrdynrng']
                min_sfr = max_sfr / dynrng


            gas_image = massmap
            gas_image = np.log10( gas_image )
            gas_image -= np.min( np.log10( min_sfr ) )
            gas_image /= np.max( np.log10( max_sfr ) - np.log10( min_sfr ) )
            gas_image *= 253.
            gas_image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('magma');
            image24 = my_cmap(gas_image/255.);


            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxsfrden']/kwargs['set_sfrdynrng']), np.log10(kwargs['set_maxsfrden']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(}\Sigma_\mathrm{sfr}\mathrm{/M_\odot/yr/kpc^2)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_sfr_surface_density.pdf')


#            ## convert to actual image using a colortable:
#            image_singledepth[ image_singledepth < 2.0 ] = 255.0
#            my_cmap=matplotlib.cm.get_cmap('hot');
#            image24 = my_cmap(image_singledepth/255.);
            return image24, massmap



    if weighting=='metallicity':
            massmap,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)	#,
#                **kwargs)


            metal_massmap, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm[ok] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )#,
#                **kwargs)

            metallicity_map = metal_massmap / massmap / 0.0134          # Asplund 2009 Table 4 solar value

            if ('set_maxmet' in kwargs) and ('set_metdynrng' in kwargs):
                kwargs['set_maxmet'] *= 1.0
                kwargs['set_metdynrng'] *= 1.0
                metallicity_map[ metallicity_map > kwargs['set_maxmet'] ] = kwargs['set_maxmet']
                metallicity_map[ metallicity_map < kwargs['set_maxmet'] / kwargs['set_metdynrng']  ] = kwargs['set_maxmet'] / kwargs['set_metdynrng']
                maxmet  = kwargs['set_maxmet']
                minmet  = kwargs['set_maxmet'] / kwargs['set_metdynrng']
                print(kwargs['set_metdynrng'])
            if ('set_maxden' in kwargs) and ('set_dynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxden'] ] = kwargs['set_maxden']
                massmap[ massmap < kwargs['set_maxden'] / kwargs['set_dynrng']  ] = kwargs['set_maxden'] / kwargs['set_dynrng']

            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )
            metallicity_image -= np.min( np.log10( minmet ) )
            metallicity_image /= np.log10( maxmet ) - np.log10( minmet )
            metallicity_image *= 253.
            metallicity_image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('Greys');
            image24 = my_cmap(metallicity_image/255.);


            if False:
                for iii in range(metallicity_image.shape[0]):
                    for jjj in range(metallicity_image.shape[0]):
                        r,g,b = color2dimage( metallicity_image[iii,jjj]/253., mass_image[iii,jjj], flip_intensity=True )
                        image24[iii,jjj,0:3] = [r,g,b]

            if True:	# True:
                for iii in range(metallicity_image.shape[0]):
                    for jjj in range(metallicity_image.shape[0]):
                        r,g,b = color2dimage( metallicity_image[iii,jjj]/255., metallicity_image[iii,jjj]/255.,
                                              width=0.8, shift=-0.1, rotation=-1.0, csat = 1.0, n_rot = 1.0,
                                              flip_intensity=True )
                        image24[iii,jjj,0:3] = [r,g,b]


                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                gradient_im = np.zeros( (2, 256, 3) )

                cb    = plt.figure(figsize=(6.0 , 2.0))
                cb_ax = cb.add_subplot( 1, 1, 1 )


                for index,element in enumerate(gradient[0,:]):
                    r,g,b = color2dimage( element, element,
                                              width=0.8, shift=-0.1, rotation=-1.0, csat = 1.0, n_rot = 1.0,
                                              flip_intensity=True )
                    gradient_im[0,index,:] = [  r,g,b  ]
                    gradient_im[1,index,:] = [  r,g,b  ]

                print(kwargs['set_maxmet'])
                print(kwargs['set_metdynrng'])
                cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxmet']/kwargs['set_metdynrng']), np.log10(kwargs['set_maxmet']), 0, 1 ] )
                cb_ax.yaxis.set_ticks([])
                cb_ax.xaxis.set_ticks([ -0.8, -0.4, 0.0, 0.4  ])
                cb_ax.xaxis.tick_top()
                cb_ax.set_xlabel('$\mathrm{Log(Z/Z_\odot)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
                cb_ax.xaxis.set_label_position('top')
                cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
                cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
                cb_ax.tick_params(axis='x', pad=3)

                cb_ax.xaxis.set_label_coords(0.5, 1.6)
                cb.savefig('./plots/cbar_met.pdf')
            return image24, metal_massmap




    if weighting=='OxygenToNitrogen':
            oxygen_mass_map,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=zm9[ok,4] * m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)

            nitrogen_mass_map, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm9[ok,3] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )

            metallicity_map = (oxygen_mass_map/16.0) / (nitrogen_mass_map/14.0) #abundance ratio

            # Solar Abundance Definition (Asplund 2009)
            # Log(O/H) + 12 = 8.69 -> [O/H] = -3.31
            # Log(N/H) + 12 = 7.83 -> [N/H] = -4.17
            # Log(O/N) = Log(O/H) - Log(N/H) = 0.86
            #
            # O/N abundence ratio:
            #     O/N = 7.244
            #

            metallicity_map /= 7.244

            print("OxygenToNitrogen Map Statistics ( before clipping ):" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( np.min(metallicity_map), np.max(metallicity_map), np.median(metallicity_map)) )

            if ('set_maxon' in kwargs) and ('set_ondynrng' in kwargs):
                metallicity_map[ metallicity_map > kwargs['set_maxon'] ] = kwargs['set_maxon']
                metallicity_map[ metallicity_map < kwargs['set_maxon'] / kwargs['set_ondynrng']  ] = \
                                                        kwargs['set_maxon'] / kwargs['set_ondynrng']
            else:
                set_maxon = 25.0
                set_ondynrng = 25.0/10.0
                metallicity_map[ metallicity_map > set_maxon ] = set_maxon
                metallicity_map[ metallicity_map < set_maxon / set_ondynrng ] = \
                                                        set_maxon / set_ondynrng
                kwargs['set_maxon'] = set_maxon
                kwargs['set_ondynrng'] = set_ondynrng

            print("OxygenToNitrogen Map Statistics ( after clipping ):" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( np.min(metallicity_map), np.max(metallicity_map), np.median(metallicity_map)) )


            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )

            print("After Log Scale OxygenToNitrogen Map Statistics:" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( 
                    np.min(metallicity_image), np.max(metallicity_image), np.median(metallicity_image))  )

            metallicity_image -= np.log10( kwargs['set_maxon'] / kwargs['set_ondynrng'] )
            metallicity_image /= np.log10( kwargs['set_ondynrng'] )
            metallicity_image *= 253.
            metallicity_image += 1.0

            print("After Scaling, metallicity_image Map Statistics:" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( \
                    np.min(metallicity_image), np.max(metallicity_image), np.median(metallicity_image)) )


            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('afmhot');
            image24 = my_cmap(metallicity_image/255.);



            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxon']/kwargs['set_ondynrng']), np.log10(kwargs['set_maxon']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(O/N)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_on_ratio.pdf')


    if weighting=='IronToOxygen':
            iron_mass_map,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=zm9[ok,8] * m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)

            oxygen_mass_map, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm9[ok,4] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )

            metallicity_map = (iron_mass_map/56.0) / (oxygen_mass_map/16.0)

            # Solar Abundance Definition (Asplund 2009)
            # Log(Fe/H)+ 12 = 7.50 -> [Fe/H]= -4.5
            # Log(O/H) + 12 = 8.69 -> [O/H] = -3.31
            # Log(Fe/O) = Log(Fe/H) - Log(O/H) = -1.19
            #
            # O/N abundence ratio:
            #     O/N = 0.0645
            #

            metallicity_map /= 0.0645

            print("IronToOxygen Map Statistics ( before clipping ):" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( np.min(metallicity_map), np.max(metallicity_map), np.median(metallicity_map)) )

            if ('set_maxfo' in kwargs) and ('set_fodynrng' in kwargs):
                metallicity_map[ metallicity_map > kwargs['set_maxfo'] ] = kwargs['set_maxfo']
                metallicity_map[ metallicity_map < kwargs['set_maxfo'] / kwargs['set_fodynrng']  ] = \
                                                        kwargs['set_maxfo'] / kwargs['set_fodynrng']
            else:
                set_maxfo = 25.0
                set_fodynrng = 25.0/10.0
                metallicity_map[ metallicity_map > set_maxfo ] = set_maxfo
                metallicity_map[ metallicity_map < set_maxfo / set_fodynrng ] = \
                                                        set_maxfo / set_fodynrng
                kwargs['set_maxfo'] = set_maxfo
                kwargs['set_fodynrng'] = set_fodynrng


            print("IronToOxygen Map Statistics:" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( np.min(metallicity_map), np.max(metallicity_map), np.median(metallicity_map)) )

            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )

            print("After Log Scale IronToOxygen Map Statistics:" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format(
                    np.min(metallicity_image), np.max(metallicity_image), np.median(metallicity_image)) )

            metallicity_image -= np.log10( kwargs['set_maxfo'] / kwargs['set_fodynrng'] )
            metallicity_image /= np.log10( kwargs['set_fodynrng'] )
            metallicity_image *= 253.
            metallicity_image += 1.0

            print("After Scaling, metallicity_image Map Statistics:" )
            print("    min map val = {:16.8f}\n    max map val = {:16.8f}\n    med map val = {:16.8f}".format( \
                    np.min(metallicity_image), np.max(metallicity_image), np.median(metallicity_image)) )


            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('bone');
            image24 = my_cmap(metallicity_image/255.);



            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxfo']/kwargs['set_fodynrng']), np.log10(kwargs['set_maxfo']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(Fe/O)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_feo_ratio.pdf')


    if True:	#else:
        return image24, oxygen_mass_map;



def illustris_stellar_image( snapdir, snapnum, fof_num, sub_num,
                         xrange=0,yrange=0,zrange=0, pixels=64, include_lighting=1,
                         weighting='stellar_mass', face_on=False, edge_on=False,
                         little_h = 0.6774,
                         **kwargs  ):


    u, rho, h, nh, ne, sfr, lx, zm, m, x, y, z, vx, vy, vz, c, time = \
        load_illustris_brick( snapdir, snapnum, fof_num, sub_num )

    m_stars, h_stars, x_stars, y_stars, z_stars, vx_stars, vy_stars, vz_stars, c_main, z_main, time = \
        load_illustris_brick( snapdir, snapnum, fof_num, sub_num, do_stars=True )


    run = snapdir[snapdir.index('Runs/')+5:snapdir.index('output')-1]
    image_center = readsubf.subfind_catalog(  snapdir, snapnum, subcat=True, grpcat=True,
                       keysel=['SubhaloPos', 'GroupFirstSub'] )

    if(fof_num >= 0):
        GroupFirstSub = image_center.GroupFirstSub[fof_num]
        if (sub_num >= 0):
            sub_num += GroupFirstSub
        else:
            sub_num = GroupFirstSub




    center = image_center.SubhaloPos[sub_num, :]
    x -= center[0]
    y -= center[1]
    z -= center[2]


    x_stars -= center[0]
    y_stars -= center[1]
    z_stars -= center[2]

    if face_on==True or edge_on==True:
        r = np.sqrt( x**2 + y**2 + z**2 )
        index = (sfr > 0) & (r < 5.0)
        lz = np.sum( m[index] * (x[index] * vy[index] - y[index] * vx[index] ) )
        lx = np.sum( m[index] * (y[index] * vz[index] - z[index] * vy[index] ) )
        ly = np.sum( m[index] * (z[index] * vx[index] - x[index] * vz[index] ) )

        if face_on:
            phi   = np.arctan2( ly, lx )    #  + 3.14159/2.0
            theta =  np.arctan2( np.sqrt(lx**2 + ly**2), lz )      # + 3.14159/2.0
        if edge_on:
            phi   = np.arctan2( ly, lx ) + 3.14159/2.0
            theta = 3.14159/2.0 + np.arctan2( np.sqrt(lx**2 + ly**2), lz )	# + 3.14159/2.0


        x_ = -z_stars    * np.sin(theta) + ( x_stars * np.cos(phi) + y_stars  * np.sin(phi)) * np.cos(theta)
        y_ = -x_stars    * np.sin(phi)   +   y_stars * np.cos(phi)
        z_ =  z_stars    * np.cos(theta) + ( x_stars * np.cos(phi) + y_stars  * np.sin(phi)) * np.sin(theta)
        vx_ = -vz_stars  * np.sin(theta) + (vx_stars * np.cos(phi) + vy_stars * np.sin(phi)) * np.cos(theta)
        vy_ = -vx_stars  * np.sin(phi)   +  vy_stars * np.cos(phi)
        vz_ =  vz_stars  * np.cos(theta) + (vx_stars * np.cos(phi) + vy_stars * np.sin(phi)) * np.sin(theta)

        x_stars = x_
        y_stars = y_
        z_stars = z_
        vx_stars = vx_
        vy_stars = vy_
        vz_stars = vz_

    # trim particles to those inside/near the plotting region
    sidebuf=10.;

    xr=xrange; yr=yrange; zr=zrange;
    if xr==0 and yr==0 and zr==0:
        xr=[np.min( x), np.max(x)]
        yr=[np.min( y), np.max(y)]
        zr=[np.min( z), np.max(z)]

    xlen=0.5*(xr[1]-xr[0]); ylen=0.5*(yr[1]-yr[0]);
    if (checklen(yrange)>1): yr=yrange;
    scale=0.5*np.max([xr[1]-xr[0],yr[1]-yr[0]]);
    if (checklen(zrange)>1): zr=zrange;
    else:  zr=np.array([-1.,1.])*scale;

    ok =    ok_scan(x_stars,xmax=np.max(np.fabs(xr))*sidebuf) & ok_scan(y_stars,xmax=np.max(np.fabs(yr))*sidebuf) & \
            ok_scan(z_stars,xmax=np.max(np.fabs(zr))*sidebuf) & ok_scan(m_stars,pos=1,xmax=1.0e40) & \
            ok_scan(h_stars,pos=1)

    cbar_fontsize=40
    if weighting=='stellar_mass':
            massmap,image_singledepth = \
            cmakepic.simple_makepic(
                 x_stars[ok],y_stars[ok],\
                 weights=m_stars[ok],hsml=h_stars[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels)
            massmap = massmap * 1e10 / 1e6 * little_h # solar masses  per cpc^2

            if ('set_maxden' in kwargs) and ('set_dynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxden'] ] = kwargs['set_maxden']
                massmap[ massmap < kwargs['set_maxden'] / kwargs['set_dynrng']  ] = kwargs['set_maxden'] / kwargs['set_dynrng']
                maxden = kwargs['set_maxden']
                dynrng = kwargs['set_dynrng']
                minden = maxden / dynrng

            image = massmap
            image = np.log10( image )
            image -= np.min( np.log10( minden ) )
            image /= np.max( np.log10(maxden) - np.log10( minden ) )
            image *= 253.
            image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('Greys');
            image24 = my_cmap(image/255.);

            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxden']/kwargs['set_dynrng']), np.log10(kwargs['set_maxden']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(}\Sigma_*\mathrm{/[M_\odot\//pc^2])}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.10, right=0.90, top=0.333, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)
            cb_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            cb_ax.xaxis.set_label_coords(0.5, 2.4)
            cb.savefig('./plots/cbar_stellar_surface_density.pdf')

            return image24, massmap


#def center_data( data, **kwargs):
#    # if "center" is set, this overrules everything else.
#    if (kwargs.get("center") is not None) and (kwargs.get("center") is not [0,0,0]):
#        for ijk in range(3):  data.pos[:,ijk] -= center[ijk]
#        return data
#
#    if (kwargs.get("cosmo") is not None) and (kwargs.get("cosmo")>0):
#        if ((kwargs.get("fof_num") is None) and (kwargs.get("sub_num") is None) ): # neither set.  Center on middle of full box.
#            boxsize = data.header.boxsize
#            for ijk in range(3):  data.pos[:,ijk] -= boxsize/2.0
#            return data
#        else:   # we are loading a specific fof and/or subhalo
#            if (kwargs.get("fof_num") is not None): fof_num = kwargs.get("fof_num")
#            else: fof_num=-1
#            if (kwargs.get("sub_num") is not None): sub_num = kwargs.get("sub_num")
#            else: sub_num=-1
#
#            subfind_cat = readsubf.subfind_catalog(  data.snapdir, data.snapnum, subcat=True, grpcat=True,
#                       keysel=['SubhaloPos', 'GroupFirstSub'] )
#
#            if(fof_num >= 0):
#                GroupFirstSub = subfind_cat.GroupFirstSub[fof_num]
#                if (sub_num >= 0):
#                    sub_num += GroupFirstSub
#                else:
#                    sub_num = GroupFirstSub
#
 #           center = subfind_cat.SubhaloPos[sub_num, :]
 #           for ijk in range(3):  data.pos[:,ijk] -= center[ijk]

 #   return data

# image types:
#     cosmo = {0,1}   -> cosmological snapshot or not
#     stellar = {0,1} -> project stellar content

# image_types are:
#   1 - gas surface density
#   2 - 

def image_maker_v2( snapdir, snapnum, parttype,
                    weight_type='mass', pixels=64, 
                    **kwargs  ):

    snapshot = naming.return_snap_filebase( snapdir, snapnum, **kwargs )[0]       # this function should be updated to always, 
                                                                        	  # generically, point to the file base of the 
                                                                        	  # relevant file

    data = rs.snapshot_data( snapshot, parttype, **kwargs ) # possible tags: cosmo, fof_num, sub_num
    data.snapdir = snapdir
    data.snapnum = snapnum 

    print("... centering data ..." )
    data = idm.center_data( data, **kwargs )


    phi, theta = util.determine_rotation_angle( data, **kwargs )
    print("... rotating data ...")
    data = idm.rotate_data( data, phi, theta )

    print("... image bounds ...")
    data = ip.determine_image_bounds( data, **kwargs )

    print("... image stretch ...")
    data = ip.determine_image_stretch( data, **kwargs )

    print("... clipping ...")
    data = idm.clip_particles( data,  data.xr, data.yr, data.zr,  **kwargs )

    print("... rescaling ...")
    data = idm.rescale_hsml(   data, **kwargs)	

    if weight_type is 'gas_dens_temp':
            gas_map_temperature_cuts=np.array([300., 2.0e4, 3.0e5 ])
            kernel_widths=np.array([0.8,0.3,0.6])
            #kernel_widths=np.array([0.7,0.3,0.7])

            # layer below is for zoom_dw large-scale GAS sims ONLY
            #gas_map_temperature_cuts=np.array([300., 1.0e4, 1.0e5 ])
            #kernel_widths=np.array([0.5,0.25,0.6])

            out_gas,out_u,out_g,out_r = rayproj.gas_raytrace_temperature( \
                gas_map_temperature_cuts, \
                data.pos[:,0], data.pos[:,1], data.pos[:,2], data.temp, data.mass, data.h, \
                xrange=data.xr, yrange=data.yr, zrange=data.zr, pixels=pixels, \
                isosurfaces = 1, kernel_width=kernel_widths, \
                add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]) );

            if np.sum(out_gas)==-1:             # there is nothing here.  zero out images and massmap.
                image = np.zeros( (pixels,pixels, 4) )
                image[:,:,3] = 1.0
                massmap = np.zeros( (pixels,pixels) )
            else:                               # use nice threeeband image bandmaps to convert data to image.
                image, massmap = \
                    makethreepic.make_threeband_image_process_bandmaps( out_r,out_g,out_u, \
                    maxden=data.set_maxden,dynrange=data.set_dynrng,pixels=pixels, \
                    color_scheme_nasa=True,color_scheme_sdss=False);

                image = makethreepic.layer_band_images(image, massmap);


    elif weight_type is not 'light':
        massmap,image = \
                cmakepic.simple_makepic(data.pos[:,0],data.pos[:,1],weights=getattr(data,weight_type),hsml=data.h,\
                xrange=data.xr,yrange=data.yr,
                set_maxden=data.set_maxden , set_dynrng=data.set_dynrng,
                pixels=pixels )

    else:
        # this is a stellar image.  If we want dust lanes, need gas too.
        gas_data = rs.snapshot_data( snapshot, 0, **kwargs ) # possible tags: cosmo, fof_num, sub_num
        gas_data.snapdir = snapdir
        gas_data.snapnum = snapnum 

        gas_data = idm.center_data( gas_data, **kwargs )
        gas_data = idm.rotate_data( gas_data, phi, theta )
        gas_data = idm.clip_particles( gas_data,  data.xr, data.yr, data.zr,  **kwargs )
        gas_data = idm.rescale_hsml( gas_data, **kwargs)	#gas_data.h * h_rescale_factor

        data = ip.set_band_ids( data, **kwargs )
        data = ip.set_kappa_units( data, **kwargs )
        data = ip.set_imf( data, **kwargs )
        data = ip.set_dust_to_gas_ratio( data, **kwargs )

        out_gas,image_u,image_g,image_r = rayproj.stellar_raytrace( data.BAND_IDS, \
                    data.pos[:,0], data.pos[:,1], data.pos[:,2], 				#x[ok], y[ok], z[ok], \
                    data.mass, data.age, data.z, data.h,					#stellar_mass[ok], stellar_age[ok], stellar_metallicity[ok], h_main[ok], \
                    gas_data.pos[:,0], gas_data.pos[:,1], gas_data.pos[:,2], gas_data.mass, \
                    gas_data.z*data.dust_to_gas_ratio_rescale, gas_data.h*2.5, \
                    xrange=data.xr, yrange=data.yr, zrange=data.zr, pixels=pixels, \
                    ADD_BASE_METALLICITY=0.1*0.02, ADD_BASE_AGE=0.0003,
                    KAPPA_UNITS=data.kappa_units,	#2.08854068444 , #* 1e-6,
                    IMF_SALPETER=data.imf_salpeter,	#0, 
                    IMF_CHABRIER=data.imf_chabrier);	#1 );

        if(np.array(out_gas).size<=1):
                    ## we got a bad return from the previous routine, initialize blank arrays
                    out_gas=image_u=image_g=image_r=np.zeros((pixels,pixels))
                    image24=massmap=np.zeros((pixels,pixels,3))
        else:
                    image, massmap = \
                        makethreepic.make_threeband_image_process_bandmaps( image_r,image_g,image_u, \
                        maxden=data.set_maxden,dynrange=data.set_dynrng,pixels=pixels, \
                        color_scheme_nasa=True,color_scheme_sdss=False );

    return image, massmap



def illustris_gas_image( snapdir, snapnum, fof_num, sub_num,
                         xrange=0,yrange=0,zrange=0, pixels=64, include_lighting=1,
                         weighting='gas', face_on=False, edge_on=False,
                         little_h = 0.6774,
                         **kwargs  ):

    # load and center data

    if weighting=='OxygenToNitrogen' or weighting=='IronToOxygen':
        u, rho, h, nh, ne, sfr, lx, zm, m, x, y, z, vx, vy, vz, c, zm9, time = load_illustris_brick( snapdir, snapnum, fof_num, sub_num, load_all_elements=True )
    else:
        u, rho, h, nh, ne, sfr, lx, zm, m, x, y, z, vx, vy, vz, c, time = load_illustris_brick( snapdir, snapnum, fof_num, sub_num )

    run = snapdir[snapdir.index('Runs/')+5:snapdir.index('output')-1]
    image_center = readsubf.subfind_catalog(  snapdir, snapnum, subcat=True, grpcat=True,
                       keysel=['SubhaloPos', 'GroupFirstSub'] )


    if(fof_num >= 0):
        GroupFirstSub = image_center.GroupFirstSub[fof_num]
        if (sub_num >= 0):
            sub_num += GroupFirstSub
        else:
            sub_num = GroupFirstSub

    center = image_center.SubhaloPos[sub_num, :]
    x -= center[0]
    y -= center[1]
    z -= center[2]

    if face_on==True or edge_on==True:
        r = np.sqrt( x**2 + y**2 + z**2 )
        index = (sfr > 0) & (r < 5.0)
        lz = np.sum( m[index] * (x[index] * vy[index] - y[index] * vx[index] ) )
        lx = np.sum( m[index] * (y[index] * vz[index] - z[index] * vy[index] ) )
        ly = np.sum( m[index] * (z[index] * vx[index] - x[index] * vz[index] ) )

        if face_on:
            phi   = np.arctan2( ly, lx )    #  + 3.14159/2.0
            theta =  np.arctan2( np.sqrt(lx**2 + ly**2), lz )      # + 3.14159/2.0
        if edge_on:
            phi   = np.arctan2( ly, lx ) + 3.14159/2.0
            theta = 3.14159/2.0 + np.arctan2( np.sqrt(lx**2 + ly**2), lz )	# + 3.14159/2.0


        x_ = -z  * np.sin(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.cos(theta)
        y_ = -x  * np.sin(phi)   + y  * np.cos(phi)
        z_ =  z  * np.cos(theta) + (x * np.cos(phi) + y *np.sin(phi)) * np.sin(theta)
        vx_ = -vz  * np.sin(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.cos(theta)
        vy_ = -vx  * np.sin(phi)   + vy  * np.cos(phi)
        vz_ =  vz  * np.cos(theta) + (vx * np.cos(phi) + vy *np.sin(phi)) * np.sin(theta)

        x = x_
        y = y_
        z = z_
        vx = vx_
        vy = vy_
        vz = vz_

#	x, y, z = coordinates_rotate(x, y, z, theta, phi, coordinates_cylindrical=0)
#        vx,vy,vz= coordinates_rotate(vx,vy,vz,theta, phi, coordinates_cylindr:ical=0)

        lz = np.sum( x * vy - y * vx )
        lx = np.sum( y * vz - z * vy )
        ly = np.sum( z * vx - x * vz )



    # trim particles to those inside/near the plotting region
    sidebuf=10.;

    xr=xrange; yr=yrange; zr=zrange;
    if xr==0 and yr==0 and zr==0:
        xr=[np.min( x), np.max(x)]
        yr=[np.min( y), np.max(y)]
        zr=[np.min( z), np.max(z)]

    xlen=0.5*(xr[1]-xr[0]); ylen=0.5*(yr[1]-yr[0]);
    if (checklen(yrange)>1): yr=yrange;
    scale=0.5*np.max([xr[1]-xr[0],yr[1]-yr[0]]);
    if (checklen(zrange)>1): zr=zrange;
    else:  zr=np.array([-1.,1.])*scale;

    ok =    ok_scan(x,xmax=np.max(np.fabs(xr))*sidebuf) & ok_scan(y,xmax=np.max(np.fabs(yr))*sidebuf) & \
            ok_scan(z,xmax=np.max(np.fabs(zr))*sidebuf) & ok_scan(m,pos=1,xmax=1.0e40) & \
            ok_scan(h,pos=1) & ok_scan(c,pos=1,xmax=1.0e40)

    cbar_fontsize=40
    if weighting=='gas':
            massmap,image_singledepth = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels)#,
               # **kwargs)
            massmap = massmap * 1e10 / 1e6 * little_h # solar masses  per cpc^2

            if ('set_maxden' in kwargs) and ('set_dynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxden'] ] = kwargs['set_maxden']
                massmap[ massmap < kwargs['set_maxden'] / kwargs['set_dynrng']  ] = kwargs['set_maxden'] / kwargs['set_dynrng']
                maxden = kwargs['set_maxden']
                dynrng = kwargs['set_dynrng']
                minden = maxden / dynrng

            gas_image = massmap
            gas_image = np.log10( gas_image )
            gas_image -= np.min( np.log10( minden ) )
            gas_image /= np.max( np.log10(maxden) - np.log10( minden ) )
            gas_image *= 253.
            gas_image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('hot');
            image24 = my_cmap(gas_image/255.);


            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxden']/kwargs['set_dynrng']), np.log10(kwargs['set_maxden']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(}\Sigma_\mathrm{gas}\mathrm{/[M_\odot\//pc^2])}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.10, right=0.90, top=0.333, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)
            cb_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            cb_ax.xaxis.set_label_coords(0.5, 2.4)
            cb.savefig('./plots/cbar_gas_surface_density.pdf')

            return image24, massmap


    if weighting=='sfr':
            if np.sum(sfr[ok]) > 0.1:            # SFR of at least 0.1 solar masses per year
                massmap,image_singledepth = \
                cmakepic.simple_makepic(x[ok],y[ok],weights=sfr[ok],hsml=h[ok],\
                    xrange=xr,yrange=yr,
                    pixels=pixels)#,
                   # **kwargs)
                massmap = massmap * little_h * little_h # solar masses per year per ckpc^2
            else:
                massmap = np.zeros( (pixels, pixels) )

            if ('set_maxsfrden' in kwargs) and ('set_sfrdynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxsfrden'] ] = kwargs['set_maxsfrden']
                massmap[ massmap < kwargs['set_maxsfrden'] / kwargs['set_sfrdynrng']  ] = kwargs['set_maxsfrden'] / kwargs['set_sfrdynrng']
                max_sfr = kwargs['set_maxsfrden']
                dynrng  = kwargs['set_sfrdynrng']
                min_sfr = max_sfr / dynrng


            gas_image = massmap
            gas_image = np.log10( gas_image )
            gas_image -= np.min( np.log10( min_sfr ) )
            gas_image /= np.max( np.log10( max_sfr ) - np.log10( min_sfr ) )
            gas_image *= 253.
            gas_image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('magma');
            image24 = my_cmap(gas_image/255.);


            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxsfrden']/kwargs['set_sfrdynrng']), np.log10(kwargs['set_maxsfrden']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(}\Sigma_\mathrm{sfr}\mathrm{/M_\odot/yr/kpc^2)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.1, right=0.90, top=0.333, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            cb_ax.xaxis.set_label_coords(0.5, 2.4)
            cb.savefig('./plots/cbar_sfr_surface_density.pdf')


#            ## convert to actual image using a colortable:
#            image_singledepth[ image_singledepth < 2.0 ] = 255.0
#            my_cmap=matplotlib.cm.get_cmap('hot');
#            image24 = my_cmap(image_singledepth/255.);
            return image24, massmap



    if weighting=='metallicity':
            massmap,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)	#,
#                **kwargs)


            metal_massmap, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm[ok] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )#,
#                **kwargs)

            metallicity_map = metal_massmap / massmap / 0.0134          # Asplund 2009 Table 4 solar value

            if ('set_maxmet' in kwargs) and ('set_metdynrng' in kwargs):
                kwargs['set_maxmet'] *= 1.0
                kwargs['set_metdynrng'] *= 1.0
                metallicity_map[ metallicity_map > kwargs['set_maxmet'] ] = kwargs['set_maxmet']
                metallicity_map[ metallicity_map < kwargs['set_maxmet'] / kwargs['set_metdynrng']  ] = kwargs['set_maxmet'] / kwargs['set_metdynrng']
                maxmet  = kwargs['set_maxmet']
                minmet  = kwargs['set_maxmet'] / kwargs['set_metdynrng']
            if ('set_maxden' in kwargs) and ('set_dynrng' in kwargs):
                massmap[ massmap > kwargs['set_maxden'] ] = kwargs['set_maxden']
                massmap[ massmap < kwargs['set_maxden'] / kwargs['set_dynrng']  ] = kwargs['set_maxden'] / kwargs['set_dynrng']

            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )
            metallicity_image -= np.min( np.log10( minmet ) )
            metallicity_image /= np.log10( maxmet ) - np.log10( minmet )
            metallicity_image *= 253.
            metallicity_image += 1.0

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('Greys');
            image24 = my_cmap(metallicity_image/255.);


            if False:
                for iii in range(metallicity_image.shape[0]):
                    for jjj in range(metallicity_image.shape[0]):
                        r,g,b = color2dimage( metallicity_image[iii,jjj]/253., mass_image[iii,jjj], flip_intensity=True )
                        image24[iii,jjj,0:3] = [r,g,b]

            if True:	# True:
                for iii in range(metallicity_image.shape[0]):
                    for jjj in range(metallicity_image.shape[0]):
                        r,g,b = color2dimage( metallicity_image[iii,jjj]/255., metallicity_image[iii,jjj]/255.,
                                              width=0.8, shift=-0.1, rotation=-1.0, csat = 1.0, n_rot = 1.0,
                                              flip_intensity=True )
                        image24[iii,jjj,0:3] = [r,g,b]


                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                gradient_im = np.zeros( (2, 256, 3) )

                cb    = plt.figure(figsize=(6.0 , 2.0))
                cb_ax = cb.add_subplot( 1, 1, 1 )


                for index,element in enumerate(gradient[0,:]):
                    r,g,b = color2dimage( element, element,
                                              width=0.8, shift=-0.1, rotation=-1.0, csat = 1.0, n_rot = 1.0,
                                              flip_intensity=True )
                    gradient_im[0,index,:] = [  r,g,b  ]
                    gradient_im[1,index,:] = [  r,g,b  ]

                cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxmet']/kwargs['set_metdynrng']), np.log10(kwargs['set_maxmet']), 0, 1 ] )
                cb_ax.yaxis.set_ticks([])
                cb_ax.xaxis.set_ticks([ -0.5, 0.0,  0.5  ])
                cb_ax.xaxis.tick_top()
                cb_ax.set_xlabel('$\mathrm{Log(Z/Z_\odot)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
                cb_ax.xaxis.set_label_position('top')
                cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
                cb.subplots_adjust(left=0.1, right=0.9, top=0.333, bottom=0.05, wspace=0.0, hspace=0.0)
                cb_ax.tick_params(axis='x', pad=3)

#                cb_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
                cb_ax.xaxis.set_label_coords(0.5, 2.4)
                cb.savefig('./plots/cbar_met.pdf')
            return image24, metallicity_map




    if weighting=='OxygenToNitrogen':
            oxygen_mass_map,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=zm9[ok,4] * m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)

            nitrogen_mass_map, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm9[ok,3] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )

            metallicity_map = (oxygen_mass_map/16.0) / (nitrogen_mass_map/14.0) #abundance ratio

            # Solar Abundance Definition (Asplund 2009)
            # Log(O/H) + 12 = 8.69 -> [O/H] = -3.31
            # Log(N/H) + 12 = 7.83 -> [N/H] = -4.17
            # Log(O/N) = Log(O/H) - Log(N/H) = 0.86
            #
            # O/N abundence ratio:
            #     O/N = 7.244
            #

            metallicity_map /= 7.244

            if ('set_maxon' in kwargs) and ('set_ondynrng' in kwargs):
                metallicity_map[ metallicity_map > kwargs['set_maxon'] ] = kwargs['set_maxon']
                metallicity_map[ metallicity_map < kwargs['set_maxon'] / kwargs['set_ondynrng']  ] = \
                                                        kwargs['set_maxon'] / kwargs['set_ondynrng']
            else:
                set_maxon = 25.0
                set_ondynrng = 25.0/10.0
                metallicity_map[ metallicity_map > set_maxon ] = set_maxon
                metallicity_map[ metallicity_map < set_maxon / set_ondynrng ] = \
                                                        set_maxon / set_ondynrng
                kwargs['set_maxon'] = set_maxon
                kwargs['set_ondynrng'] = set_ondynrng

            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )

            metallicity_image -= np.log10( kwargs['set_maxon'] / kwargs['set_ondynrng'] )
            metallicity_image /= np.log10( kwargs['set_ondynrng'] )
            metallicity_image *= 253.
            metallicity_image += 1.0



            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('afmhot');
            image24 = my_cmap(metallicity_image/255.);



            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxon']/kwargs['set_ondynrng']), np.log10(kwargs['set_maxon']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(O/N)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_on_ratio.pdf')


    if weighting=='IronToOxygen':
            iron_mass_map,image_dummy1 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=zm9[ok,8] * m[ok],hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30)

            oxygen_mass_map, image_dummy2 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=(zm9[ok,4] * m[ok]),hsml=h[ok],\
                xrange=xr,yrange=yr,
                pixels=pixels,
                set_maxden=1e10,
                set_dynrng=1e30 )

            metallicity_map = (iron_mass_map/56.0) / (oxygen_mass_map/16.0)

            # Solar Abundance Definition (Asplund 2009)
            # Log(Fe/H)+ 12 = 7.50 -> [Fe/H]= -4.5
            # Log(O/H) + 12 = 8.69 -> [O/H] = -3.31
            # Log(Fe/O) = Log(Fe/H) - Log(O/H) = -1.19
            #
            # O/N abundence ratio:
            #     O/N = 0.0645
            #

            metallicity_map /= 0.0645

            if ('set_maxfo' in kwargs) and ('set_fodynrng' in kwargs):
                metallicity_map[ metallicity_map > kwargs['set_maxfo'] ] = kwargs['set_maxfo']
                metallicity_map[ metallicity_map < kwargs['set_maxfo'] / kwargs['set_fodynrng']  ] = \
                                                        kwargs['set_maxfo'] / kwargs['set_fodynrng']
            else:
                set_maxfo = 25.0
                set_fodynrng = 25.0/10.0
                metallicity_map[ metallicity_map > set_maxfo ] = set_maxfo
                metallicity_map[ metallicity_map < set_maxfo / set_fodynrng ] = \
                                                        set_maxfo / set_fodynrng
                kwargs['set_maxfo'] = set_maxfo
                kwargs['set_fodynrng'] = set_fodynrng


            metallicity_image = metallicity_map
            metallicity_image = np.log10( metallicity_image )

            metallicity_image -= np.log10( kwargs['set_maxfo'] / kwargs['set_fodynrng'] )
            metallicity_image /= np.log10( kwargs['set_fodynrng'] )
            metallicity_image *= 253.
            metallicity_image += 1.0


            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('bone');
            image24 = my_cmap(metallicity_image/255.);



            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            gradient_im = np.zeros( (2, 256, 4) )

            cb    = plt.figure(figsize=(6.0 , 2.0))
            cb_ax = cb.add_subplot( 1, 1, 1 )

            for index,element in enumerate(gradient[0,:]):
                r,g,b,a = my_cmap( element )
                gradient_im[0,index,:] = [  r,g,b,a  ]
                gradient_im[1,index,:] = [  r,g,b,a  ]

            cb_ax.imshow(gradient_im, aspect='auto', extent=[ np.log10(kwargs['set_maxfo']/kwargs['set_fodynrng']), np.log10(kwargs['set_maxfo']), 0, 1 ] )
            cb_ax.yaxis.set_ticks([])
            cb_ax.xaxis.tick_top()
            cb_ax.set_xlabel('$\mathrm{Log(Fe/O)}$', fontsize=cbar_fontsize)   #r'$\mathrm{Log(Pressure\/[Code\/Units])}$')
            cb_ax.xaxis.set_label_position('top')
            cb_ax.tick_params(axis='both', which='major', labelsize=cbar_fontsize) # was 25
            cb.subplots_adjust(left=0.05, right=0.95, top=0.5, bottom=0.05, wspace=0.0, hspace=0.0)
            cb_ax.tick_params(axis='x', pad=3)

            cb_ax.xaxis.set_label_coords(0.5, 1.6)
            cb.savefig('./plots/cbar_feo_ratio.pdf')


    if True:	#else:
        return image24, oxygen_mass_map;


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    norm, A, mu, sigma = p
    return norm + A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

def image_maker( sdir, snapnum, \
    snapdir_master='',	#/n/scratch2/hernquist_lab/phopkins/sbw_tests/',
    outdir_master='',	#/n/scratch2/hernquist_lab/phopkins/images/',
    theta=0., phi=0., dynrange=1.0e5, maxden_rescale=1., maxden=0.,
    show_gasstarxray = 'gas', #	or 'star' or 'xray'
    add_gas_layer_to_image='', set_added_layer_alpha=0.3, set_added_layer_ctable='heat_purple',
    add_extension='', show_time_label=0, show_scale_label=0,
    filename_set_manually='',
    do_with_colors=1, log_temp_wt=1, include_lighting=1,
    set_percent_maxden=0, set_percent_minden=0,
    use_h0=1, cosmo=0,
    coordinates_cylindrical=0,
    pixels=720,xrange=[-1.,1.],yrange=0,zrange=0,set_temp_max=0,set_temp_min=0,
    threecolor=1, nasa_colors=1, sdss_colors=0, use_old_extinction_routine=0,
    dust_to_gas_ratio_rescale=1.0,
    project_to_camera=0, camera_opening_angle=45.0,
    center_is_camera_position=0, camera_direction=[0.,0.,-1.],
    gas_map_temperature_cuts=[1.0e4, 1.0e6],
    input_data_is_sent_directly=0, \
    m_all=0,x_all=0,y_all=0,z_all=0,c_all=0,h_all=0,zm_all=0,\
    gas_u=0,gas_rho=0,gas_hsml=0,gas_numh=0,gas_nume=0,gas_metallicity=0,\
    gas_mass=0,gas_x=0,gas_y=0,gas_z=0,time=0,
    invert_colors=0,spin_movie=0,scattered_fraction=0.01,
    min_stellar_age=0,h_rescale_factor=1.,h_max=0,
    angle_to_rotate_image=0.,
    aux_maps_1=False,
    aux_maps_2=False,
    velocity_map=False, # this adds a velocity field
    pressure_map=False,
    metallicity_map=False,
    speed_map=False,
    temperature_map=False,
    density_map=False,
    skip_sfr=0,
    force_snapname=None,
    nebular_emission=False,
    illustris=False, illustrisTNG=False,fof_num=None,sub_num=None,
    **kwargs):

	## define some of the variables to be used
    ss=snap_ext(snapnum,four_char=1);
    tt=snap_ext(np.around(theta).astype(int));
    theta *= math.pi/180.; phi *= math.pi/180.; # to radians
    outputdir = outdir_master+sdir;
    call(["mkdir",outputdir]);
    outputdir+='/'; snapdir=snapdir_master+sdir;
    suff='_'+show_gasstarxray;
    if (threecolor==1):
        if (sdss_colors==1): suff+='_S3c'
        if (nasa_colors==1): suff+='_N3c'
    suff+=add_extension;
    do_xray=0; do_stars=0; do_with_sfr=0; do_with_met=0; do_gas_momentum=0;
    if((show_gasstarxray=='xr') or (show_gasstarxray=='xray')): do_xray=1; do_with_colors=0; do_with_sfr=0;
    if((show_gasstarxray=='star') or (show_gasstarxray=='stars') or (show_gasstarxray=='st')): do_stars=1; do_with_sfr=0;
    if((show_gasstarxray=='sfr') or (show_gasstarxray=='starformation')): do_xray=0; do_with_colors=0; do_with_sfr=1;
    if((show_gasstarxray=='gas_metallicity')):  do_with_colors=0; do_with_met=1;
    if(nebular_emission): do_stars=0


    ## check whether the data needs to be pulled up, or if its being passed by the calling routine
    if force_snapname is None:
        snapshot = rs.resolve_snapname( snapdir, snapnum )
    else:
        snapshot = force_snapname

    header  = rs.snapshot_header( snapshot )
    time    = header.time

    if (illustris or illustrisTNG or (cosmo>0 and (fof_num>=0 or sub_num>=0))):             #=False, illustrisTNG=False,fofnr=None,subnr=None,
        print('loading from illustris')
        if(do_stars==1):
            print('loading from stellar load routine')
            m_main, h_main, x_main, y_main, z_main, vx_main, vy_main, vz_main, c_main, zm_main, time = \
                load_illustris_brick( snapdir, snapnum, fof_num, sub_num, do_stars=True, center=True )


        else:
            print('loading illustris brick for gas')
            u_gas, rho_main, h_main, nh_dummy, nume_gas, sfr_gas, lx_dummy, zm_main, m_main, x_main, y_main, z_main, vx_main, vy_main, vz_main, c_main, time = \
            load_illustris_brick( snapdir, snapnum, fof_num, sub_num , center=True)
        if (fof_num>=0 or sub_num>=0):
            kwargs['center_pos'] = [0,0,0]
            kwargs['center_on_bh'] = False
            kwargs['center_on_cm'] = False
            


    else:
        ptypes=[2,3,4];                 ## stars in non-cosmological snapshot
        if (cosmo==1): ptypes=[4];      ## stars in cosmological snapshot
        if (do_stars==0): ptypes=[0];   ## gas, if not using stars.

        header = rs.snapshot_header( snapshot )
        mask = np.ones_like( ptypes, dtype=np.bool )
        ptypes = np.array( ptypes )
        for this_index,this_type in enumerate(ptypes):
            if header.nall[this_type] == 0:
                mask[ this_index ] = False 
        ptypes = ptypes[ mask ] 

        m_main = np.array([])
        x_main = np.array([]); y_main=np.array([]); z_main=np.array([])
        zm_main= np.array([])
        for ptype in ptypes:
            m_main   = np.append( m_main, np.array( rs.read_block( snapshot, 'MASS', ptype) ) )
            pos_main = rs.read_block( snapshot, 'POS ', ptype)
            x_main   = np.append( x_main, np.array(pos_main[:,0]) )
            y_main   = np.append( y_main, np.array(pos_main[:,1]) )
            z_main   = np.append( z_main, np.array(pos_main[:,2]) )

            if (ptype == 2) or (ptype == 3):
                zm_main = np.append(  zm_main, np.ones_like( np.array( rs.read_block( snapshot, 'MASS', ptype) ) ) * 0.01  ) 
            else:
                try:    zm_main  = np.append( zm_main, np.array( rs.read_block( snapshot, 'GZ  ', ptype) ) )
                except: zm_main  = np.append( zm_main, np.array( rs.read_block( snapshot, 'Z   ', ptype) ) )

        if do_stars==0 or aux_maps_1 or aux_maps_2:       #pressure_map or temperature_map or metallicity_map:
            rho_main = rs.read_block( snapshot, 'RHO ', 0)
            h_main   = ( m_main * 3.0 /(4.0*3.14159 * rho_main)    )**0.3333
            u_gas    = np.array( rs.read_block( snapshot, 'U   ', 0) )
            nume_gas = np.array( rs.read_block( snapshot, 'NE  ', 0) )
            c_main   = units.gas_code_to_temperature( u_gas, nume_gas )
            if aux_maps_2:
                vel_main = rs.read_block( snapshot, 'VEL ', 0)
                gas_velocity     = np.sqrt( vel_main[:,0]**2 + vel_main[:,1]**2 + vel_main[:,2]**2 )
                gas_bfield       = rs.read_block( snapshot, 'BFLD', 0)
                gas_bfield_mag   = np.sqrt( gas_bfield[:,0]**2 + gas_bfield[:,1]**2 + gas_bfield[:,2]**2 )
                gas_mach_numb    = np.array( rs.read_block( snapshot, 'MACH', 0) )

        else:
            if True:
                h_main = calc_hsml.get_particle_hsml(x_main, y_main, z_main )
            else:
                h_main = np.ones_like( x_main )
            form_time_list = np.array([])  # actually a formation time; not an age.

            for ptype in ptypes:
                if (ptype == 2) or (ptype == 3):
                    form_time_list = np.append( form_time_list, np.ones_like(    np.array( rs.read_block( snapshot, 'MASS', ptype) ) ) - 5.0 )
                else:
                    try:    form_time_list = np.append( form_time_list, np.array( rs.read_block( snapshot, 'GAGE', ptype) ) )
                    except: form_time_list = np.append( form_time_list, np.array( rs.read_block( snapshot, 'AGE ', ptype) ) )

            if cosmo==1:
                import util.cosmo_tools as ct
                z_form = 1.0/form_time_list - 1.0
                z_now  = 1.0/time - 1.0

                if False:   #nebular_emission:
                    t_form = np.zeros_like( z_form) - 0.0010
                    t_now  = 0.0            # makes all stars 10 Myrs old.
                else:
                    t_form = ct.quick_lookback_time( z_form )
                    t_now  = ct.quick_lookback_time( z_now  )
                c_main = t_now - t_form     # age of stellar population in Gyrs.

                c_main[ np.isnan( c_main ) ] = -1
                m_main[ c_main <= 0 ] = 0
            else:
                c_main = time - form_time_list


    if nebular_emission:
        threecolor = 0
        do_with_colors = 0

        line_flux_output_file = '/n/home01/ptorrey/Share/kblumenthal/nebular_emission_images/halpha_line_flux_result_array.npz'

        data = np.load( line_flux_output_file )
        line_flux = data['arr_0']
        full_age_array = data['arr_1']
        full_z_array   = data['arr_2']
        full_u_array   = data['arr_3']

        points = np.zeros( (len(full_age_array),3) )
        points[:,0] = full_age_array
        points[:,1] = full_z_array
        points[:,2] = full_u_array
        from scipy.interpolate import LinearNDInterpolator
        func = LinearNDInterpolator(  points, line_flux   )         # line strenght is L_solar / 1 M_solar

        if (illustris or illustrisTNG):
            print("sfr_gas should already be set:")
            print(sfr_gas)
        else:
            print("assume need to load the full block")
            sfr_gas = rs.read_block( snapshot, 'SFR ', 0)
        index = sfr_gas > 0

        m_main   = m_main[index] 
        x_main   = x_main[index] 
        y_main   = y_main[index]
        z_main   = z_main[index]
        zm_main  = zm_main[index]

        #rho_main = rho_main[index]
        h_main   = h_main[index]  
        #u_gas    = u_gas[index]  
        #nume_gas = nume_gas[index]
        c_main   = c_main[index]  
        sfr_gas  = sfr_gas[index]

        resample = True
        if resample:
            # assume same SFR over past 20 Myrs 
            n_age_bins = 20
            min_age = 0.001
            max_age = 0.020
            n_orig     = len(x_main)
            m_new = np.zeros( len(x_main)*n_age_bins )
            x_new = np.zeros( len(x_main)*n_age_bins )
            y_new = np.zeros( len(x_main)*n_age_bins )
            z_new = np.zeros( len(x_main)*n_age_bins )
            c_new = np.zeros( len(x_main)*n_age_bins )
            zm_new = np.zeros( len(x_main)*n_age_bins )
            age_new = np.zeros( len(x_main)*n_age_bins )
            h_new = np.zeros( len(x_main)*n_age_bins )
            sfr_new = np.zeros( len(x_main)*n_age_bins )

            for age_bin_index in range(n_age_bins):
                i1 = int(np.round( age_bin_index*n_orig))
                i2 = int(np.round((1.0+age_bin_index)*n_orig))
                m_new[ i1:i2 ] = sfr_gas * (max_age-min_age)*1e9 / n_age_bins       # this is a mass in solar masses       # m_main / (1.0*n_age_bins)
                x_new[ i1:i2 ] = x_main + np.random.randn( n_orig ) * h_main * 3.0
                y_new[ i1:i2 ] = y_main + np.random.randn( n_orig ) * h_main * 3.0
                z_new[ i1:i2 ] = z_main + np.random.randn( n_orig ) * h_main * 3.0
                zm_new[i1:i2 ]  = zm_main
                age_new[i1:i2]  = age_bin_index / (n_age_bins*1.0 - 1.0) * (max_age - min_age) + min_age
                h_new[i1:i2]    = h_main
                sfr_new[i1:i2]  = sfr_gas / (1.0*n_age_bins)
                c_new[i1:i2]  = c_main

            m_main   = m_new
            x_main   = x_new
            y_main   = y_new
            z_main   = z_new
            c_main   = c_new
            zm_main  = zm_new
            h_main   = h_new
            sfr_gas  = sfr_new
            age_main = age_new 


        line_strength = func( age_main,  zm_main, np.ones_like(zm_main)*-2.5  ) * m_main        #* 1e10 / 0.7
        line_strength[ line_strength < 0 ] = 0.0
        m_main = line_strength 
        print( "I think the total (log) line flux is {:f} L_solar".format( np.log10( np.sum(m_main) ) ) )
        print( "I think the total (log) line flux is {:f} erg/s".format(   np.log10( np.sum(m_main)*3.839e33)  ) )
        print( "The derived SFR is {:f} and the real SFR is {:f}".format(  np.sum(m_main)*3.839e33 * 7.9e-42, np.sum( sfr_gas )  ) )
        print( "The min/max of the x/y/z positions are:" )
        print( np.min( x_main ), np.max( x_main ) )
        print( np.min( y_main ), np.max( y_main ) )
        print( np.min( z_main ), np.max( z_main ) ) 
#        sys.exit()



    if(do_with_sfr==1):
        sfr_gas = rs.read_block( snapshot, 'SFR ', 0)
        m_main  = sfr_gas + 1e-6
        print("WARNING: I've loaded the the SFRs into the masses")

    if ((do_stars==1) & (threecolor==1)): ## will need gas info to process attenuation
        gas_u        = np.array( rs.read_block( snapshot, 'U   ', 0) )
        gas_rho      = np.array( rs.read_block( snapshot, 'RHO ', 0) )
        gas_mass     = np.array( rs.read_block( snapshot, 'MASS', 0) )
        gas_hsml     = ( 3.0 * gas_mass / (4.0 * np.pi * gas_rho )  )**0.3333 * 3.0
        gas_numh     = np.array( rs.read_block( snapshot, 'NH  ', 0) )
        gas_nume     = np.array( rs.read_block( snapshot, 'NE  ', 0) )
        gas_sfr      = np.array( rs.read_block( snapshot, 'SFR ', 0) )
        try:    gas_metallicity    = np.array( rs.read_block( snapshot, 'GZ  ', 0) )
        except: gas_metallicity    = np.array( rs.read_block( snapshot, 'Z   ', 0) )

        gas_pos    = np.array( rs.read_block( snapshot, 'POS ', 0) )
        gas_x      = gas_pos[:,0]
        gas_y      = gas_pos[:,1]
        gas_z      = gas_pos[:,2]

        gas_temp = gadget.gas_temperature(gas_u,gas_nume);
        gas_metallicity[gas_temp > 1.0e6] = 0.0;

        gas_x,gas_y,gas_z  = center_position( gas_x, gas_y, gas_z, **kwargs )
        gx,gy,gz           =coordinates_rotate(gas_x,gas_y,gas_z,theta,phi,coordinates_cylindrical=coordinates_cylindrical);

    h_main *= 1.25 * h_rescale_factor
    if (h_max > 0):
        h_main = np.minimum(h_main,h_max)

    # Adjust x/y/z positions to account for frame center position.
    x_main,y_main,z_main  = center_position( x_main, y_main, z_main, **kwargs )

    x,y,z    =coordinates_rotate(x_main ,y_main ,z_main ,theta,phi,coordinates_cylindrical=coordinates_cylindrical);
    try:    vx,vy,vz =coordinates_rotate(vx_all,vy_all,vz_all,theta,phi,coordinates_cylindrical=coordinates_cylindrical);
    except: print("no velocities to shift")


    ## set dynamic ranges of image
    temp_max=1.0e6; temp_min=1.0e3; ## max/min gas temperature
    if (do_stars==1): temp_max=50.; temp_min=0.01; ## max/min stellar age
    if (set_temp_max != 0): temp_max=set_temp_max
    if (set_temp_min != 0): temp_min=set_temp_min
    xr=xrange; yr=xr; zr=0;

    if (checklen(yrange)>1): yr=yrange;
    scale=0.5*np.max([xr[1]-xr[0],yr[1]-yr[0]]);

    if (checklen(zrange)>1):
        zr=zrange;
    else:
        zr=np.array([-1.,1.])*scale;
    if(maxden==0.):
        maxden = 0.1 * 1.e11/1.0e10 * (2.0/scale)**(0.3) * maxden_rescale;
    if ((threecolor==1) & (do_stars)):
        if(dynrange==1.0e5): # reset if it's the default value
            dynrange=1.0e2;
            if (nasa_colors==1): dynrange=1.0e4;
            if (nasa_colors==1): maxden *= 30.;


    ## now check if we're projecting to a camera (instead of a fixed-plane)
    xlen=0.5*(xr[1]-xr[0]); ylen=0.5*(yr[1]-yr[0]);
    xr_0=xr; yr_0=yr;
    if (project_to_camera==1):
        ## determine the angular opening and camera distance given physical x/y range:
        ## note: camera_opening_angle is the half-angle
        xr=np.array([-1.,1.])*np.tan(camera_opening_angle*math.pi/180.); yr=xr*ylen/xlen; ## n- degree opening angle
        ## use this to position camera
        c_dist = xlen/xr[1];
        camera_direction /= np.sqrt(camera_direction[0]**2.+camera_direction[1]**2.+camera_direction[2]**2.);
        camera_pos = -c_dist * camera_direction;
        ## and determine z-depth of image
        cpmax=np.sqrt(camera_pos[0]**2.+camera_pos[1]**2.+camera_pos[2]**2.); zrm=np.sqrt(zr[0]**2.+zr[1]**2.);
        ## clip z_range for speed :
        zrm=5.0*zrm;
        #zrm=2.5*zrm
        zrm=np.sqrt(zrm*zrm+cpmax*cpmax); zr=[-zrm,0.]
        ## correct if the center given is the camera position:
        if (center_is_camera_position==1): camera_pos=[0.,0.,0.];

        ## now do the actual projection into camera coordinates:
        x,y,z,rc=coordinates_project_to_camera(x,y,z,camera_pos=camera_pos,camera_dir=camera_direction);
        h_main*=rc; ## correct for size re-scaling
        m_main*=1./(z*z + h_main*h_main + (0.25*c_dist)**2.); ## makes into 'fluxes' (dimmer when further away)
        ## clip particles too close to camera to prevent artifacts
        z_clip = -c_dist/10.0
        m_main[z >= z_clip] = 0.0
        ## also need to correct gas properties if using it for attenuation

        #rotate the image; assumes angle is specified in degrees
        angle_to_rotate_image*=math.pi/180.

        xnew = x*np.cos(angle_to_rotate_image) - y*np.sin(angle_to_rotate_image)
        ynew = x*np.sin(angle_to_rotate_image) + y*np.cos(angle_to_rotate_image)
        x = xnew
        y = ynew

        if ((threecolor==1) & (do_stars==1)):
            gx,gy,gz,grc=coordinates_project_to_camera(gx,gy,gz,camera_pos=camera_pos,camera_dir=camera_direction);
            gas_hsml*=grc; gas_mass*=1./(gz*gz + gas_hsml*gas_hsml + (0.25*c_dist)**2.); ## this conserves surface density, so attenuation units are fine
            gas_mass[gz >= z_clip] = 0.0 ## clipping to prevent artifacts

            gxnew = gx*np.cos(angle_to_rotate_image) - gy*np.sin(angle_to_rotate_image)
            gynew = gx*np.sin(angle_to_rotate_image) + gy*np.cos(angle_to_rotate_image)
            gx = gxnew
            gy = gynew



    #plotting_setup_junk
    plt.close('all')
    format = '.png' ## '.ps','.pdf','.png' also work well
    axis_ratio = ylen/xlen
    #fig=plt.figure(frameon=False,figsize=(10.,10.*axis_ratio),dpi=100)
    fig=plt.figure(frameon=False,dpi=pixels)
    fig.set_size_inches(1,1.*axis_ratio)
    ax_fig=plt.Axes(fig,[0.,0.,1.,1.*axis_ratio])
    ax_fig.set_axis_off()
    fig.add_axes(ax_fig)


	## trim particles to those inside/near the plotting region
    sidebuf=10.;
    if (cosmo==1): sidebuf=1.15
    x=np.array(x,dtype='f');   y=np.array(y,dtype='f');   z=np.array(z,dtype='f');
    try:
        vx=np.array(vx,dtype='f'); vy=np.array(vy,dtype='f'); vz=np.array(vz,dtype='f');
    except:
        print("no velocities...")
    m_main=np.array(m_main,dtype='f'); h_main=np.array(h_main,dtype='f');
    c_main=np.array(c_main,dtype='f');

    ok =    ok_scan(x,xmax=np.max(np.fabs(xr))*sidebuf) & ok_scan(y,xmax=np.max(np.fabs(yr))*sidebuf) & \
            ok_scan(z,xmax=np.max(np.fabs(zr))*sidebuf) & ok_scan(m_main,pos=1,xmax=1.0e40) & \
            ok_scan(h_main,pos=1) & ok_scan(c_main,pos=1,xmax=1.0e40)

    weights = m_main; color_weights = c_main; ## recall, gadget masses in units of 1.0d10
    print("Mean/Median h_all = {:.4f}/{:.4f}".format( np.mean( h_main ) ,np.median(h_main) ) )


    ## alright, now ready for the main plot construction:

    if aux_maps_1:                                      #pressure_map or temperature_map or metallicity_map:
        t_main = units.gas_code_to_temperature( u_gas , nume_gas )					# K
        rho_main = rho_main * 1e10 * (1.989e33/3.086e21) / (1.6726e-24*3.086e21) / (3.086e21)	# cm^-3

        gas_mass,gas_pressure_map,gas_temperature_map,gas_metallicity_map = rayproj.raytrace_projection_compute( \
                x[ok], y[ok], z[ok],    \
                h_main[ok], m_main[ok],   \
                t_main[ok]*rho_main[ok]*m_main[ok], t_main[ok]*m_main[ok], zm_main[ok]*m_main[ok], \
                0.0, 0.0, 0.0,          \
                xrange=xr, yrange=yr, zrange=zr, pixels=pixels )

        gas_pressure_map = gas_pressure_map / gas_mass
        gas_pressure_map[ np.isnan( gas_pressure_map ) ] = 0.0

        gas_temperature_map = gas_temperature_map / gas_mass
        gas_temperature_map[ np.isnan( gas_temperature_map ) ] = 0.0

        gas_metallicity_map =          gas_metallicity_map / gas_mass
        gas_metallicity_map[ np.isnan( gas_metallicity_map ) ] = 0.0

        return gas_mass, gas_pressure_map, gas_temperature_map, gas_metallicity_map


    if aux_maps_2:                                      #pressure_map or temperature_map or metallicity_map:
        aux_q1 = gas_velocity
        aux_q2 = gas_bfield_mag
        aux_q3 = gas_mach_numb

        gas_mass,gas_q1_map,gas_q2_map,gas_q3_map = rayproj.raytrace_projection_compute( \
                x[ok], y[ok], z[ok],    \
                h_main[ok], m_main[ok],   \
                aux_q1[ok]*m_main[ok], aux_q2[ok]*m_main[ok], aux_q3[ok]*m_main[ok], \
                0.0, 0.0, 0.0,          \
                xrange=xr, yrange=yr, zrange=zr, pixels=pixels )

        gas_q1_map  = gas_q1_map / gas_mass
        gas_q1_map[ np.isnan( gas_q1_map ) ] = 0.0
        gas_q2_map  = gas_q2_map / gas_mass
        gas_q2_map[ np.isnan( gas_q2_map ) ] = 0.0
        gas_q3_map  = gas_q3_map / gas_mass
        gas_q3_map[ np.isnan( gas_q3_map ) ] = 0.0

        return gas_mass, gas_q1_map, gas_q2_map, gas_q3_map



    if (threecolor==1):
        if (do_stars==1):
            ## making a mock 3-color composite image here ::

            ## first grab the appropriate, attenuated stellar luminosities
            BAND_IDS=[9,10,11]  ## ugr composite
            ## for now not including BHs, so initialize some appropriate dummies:
            bh_pos=[0,0,0]; bh_luminosity=0.; include_bh=0;
            ## and some limits for the sake of integration:
            SizeMax=20.*np.max([np.fabs(xr[1]-xr[0]),np.fabs(yr[1]-yr[0])]);
            SizeMin=np.min([np.fabs(xr[1]-xr[0]),np.fabs(yr[1]-yr[0])])/250.;

            ## now set up the main variables:
            star_pos=np.zeros((3,checklen(x[ok])));
            star_pos[0,:]=x[ok]; star_pos[1,:]=y[ok]; star_pos[2,:]=z[ok];
            stellar_age=np.maximum(c_main,min_stellar_age);
            stellar_metallicity=zm_main; stellar_mass=m_main;
            print('Min stellar age = ',stellar_age.min() )
            print('Mean stellar age = ',stellar_age.mean() )
            print('Mean stellar metallicity = ',stellar_metallicity.mean() )
            ## gas will get clipped in attenuation routine, don't need to worry about it here
            gas_pos=np.zeros((3,checklen(gx)));
            gas_pos[0,:]=gx; gas_pos[1,:]=gy; gas_pos[2,:]=gz;

            if ((add_gas_layer_to_image=='')==False):
                gas_wt = 0.*gx
                maxden_for_layer=1.*maxden; dynrange_for_layer=1.*dynrange
                set_percent_maxden_layer=0.; set_percent_minden_layer=0.;
                gas_hsml_for_extra_layer=gas_hsml
                if (add_gas_layer_to_image=='Halpha'):
                    gas_wt = gas_mass * gas_rho*gas_nume*(1.-gas_numh)*(gadget.gas_temperature(gas_u,gas_nume)**(-0.75))/(gadget.gas_mu(gas_nume)**2.)
                    maxden_for_layer *= 3.e4;
                    #dynrange_for_layer *= 1.5;
                    dynrange_for_layer *= 0.7;
                    dynrange_for_layer *= 2.0;
                    #gas_hsml_for_extra_layer *= 1.1;
                if (add_gas_layer_to_image=='CO'):
                    gas_tmp = gadget.gas_temperature(gas_u,gas_nume)
                    n_tmp = gas_rho * 176.2; # assuming mean molec weight of 2.3 for dense gas
                    gas_wt = gas_mass * np.exp(-(gas_tmp/8000. + 10./n_tmp));
                    maxden_for_layer *= 3.e4;
                    dynrange_for_layer *= 0.7e4;
                if (add_gas_layer_to_image=='SFR'):
                    gas_wt = gas_sfr
                    set_percent_maxden_layer=0.9999;
                    set_percent_minden_layer=0.01;
                if (add_gas_layer_to_image=='Xray'):
                    gas_wt = gas_lxray
                    maxden_for_layer *= 0.6e3;
                    dynrange_for_layer *= 0.1;
                    gas_hsml_for_extra_layer *= 1.2;
                if (add_gas_layer_to_image=='Zmetal'):
                    gas_wt = gas_mass*gas_metallicity
                    set_percent_maxden_layer=0.9999;
                    set_percent_minden_layer=0.01;
                gas_wt /= np.sum(gas_wt)

                massmap_gas_extra_layer,image_singledepth_extra_layer = \
                cmakepic.simple_makepic(gx,gy,weights=gas_wt,hsml=gas_hsml_for_extra_layer,\
                    xrange=xr,yrange=yr,
                    set_dynrng=dynrange_for_layer,set_maxden=maxden_for_layer,
                    set_percent_maxden=set_percent_maxden_layer,set_percent_minden=set_percent_minden_layer,
                    color_temperature=0,pixels=pixels,invert_colorscale=1-invert_colors);


            if (x[ok].size <= 3):
                ## we got a bad return from the previous routine, initialize blank arrays
                out_gas=out_u=out_g=out_r=np.zeros((pixels,pixels))
                image24=massmap=np.zeros((pixels,pixels,3))
            else:
                ## actually call the ray-trace:
                out_gas,out_u,out_g,out_r = rayproj.stellar_raytrace( BAND_IDS, \
                    x[ok], y[ok], z[ok], \
                    stellar_mass[ok], stellar_age[ok], stellar_metallicity[ok], h_main[ok], \
                    gx, gy, gz, gas_mass, \
                    gas_metallicity*dust_to_gas_ratio_rescale, gas_hsml, \
                    xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
                    ADD_BASE_METALLICITY=0.1*0.02, ADD_BASE_AGE=0.0003,
                    #ADD_BASE_METALLICITY=0.001*0.02, ADD_BASE_AGE=0.0003,
                    KAPPA_UNITS=2.08854068444 , #* 1e-6,
                    IMF_SALPETER=0, IMF_CHABRIER=1 );

                if(np.array(out_gas).size<=1):
                    ## we got a bad return from the previous routine, initialize blank arrays
                    out_gas=out_u=out_g=out_r=np.zeros((pixels,pixels))
                    image24=massmap=np.zeros((pixels,pixels,3))
                else:
                    ## make the resulting maps into an image
                    print(maxden )
                    print(dynrange)
                    #sys.exit()
                    image24, massmap = \
                        makethreepic.make_threeband_image_process_bandmaps( out_r,out_g,out_u, \
                        maxden=maxden,dynrange=dynrange,pixels=pixels, \
                        color_scheme_nasa=nasa_colors,color_scheme_sdss=sdss_colors );

        else:
            ## threecolor==1, but do_stars==0, so doing a multi-pass gas image
            ##
            ##  -- many experiments here: doing gas isosurfaces with broad kernels
            ##       and overlaying a custom set of color tables after the fact seems
            ##       best. adjust opacity (kappa_units) and kernel width as needed.
            ##       also, can set 'dynrange=0' to automatically produce a given
            ##       fraction of saturated/unsaturated pixels
            ##
            gas_map_temperature_cuts=np.array([300., 2.0e4, 3.0e5 ])
            kernel_widths=np.array([0.8,0.3,0.6]) 
            #kernel_widths=np.array([0.7,0.3,0.7])

            # layer below is for zoom_dw large-scale GAS sims ONLY
            #gas_map_temperature_cuts=np.array([300., 1.0e4, 1.0e5 ])
            #kernel_widths=np.array([0.5,0.25,0.6])

            out_gas,out_u,out_g,out_r = rayproj.gas_raytrace_temperature( \
                gas_map_temperature_cuts, \
                x[ok], y[ok], z[ok], color_weights[ok], weights[ok], h_main[ok], \
                xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
                isosurfaces = 1, kernel_width=kernel_widths, \
                add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]) );
                #add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([4.1,2.0,2.0]));

            fig,ax = plt.subplots()

            if velocity_map:
                boost_val = np.min( [np.min(vx[ok]), np.min(vy[ok]), np.min(vz[ok])]   ) - 1.0
                gas_mass,gas_px,gas_py,gas_pz = rayproj.raytrace_projection_compute( \
                        x[ok], y[ok], z[ok],    \
                        h_main[ok], m_main[ok],   \
                        (vx[ok]-boost_val)*m_main[ok], (vy[ok]-boost_val)*m_main[ok], (vz[ok]-boost_val)*m_main[ok], \
                        0.0, 0.0, 0.0,          \
                        xrange=xr, yrange=yr, zrange=zr, pixels=pixels )
                tmp=np.min( gas_mass[ gas_mass > 0 ] )
                gas_mass[ gas_mass == 0 ] = tmp

                gas_vx_map = gas_px / gas_mass + boost_val;
                gas_vy_map = gas_py / gas_mass + boost_val;
                gas_vz_map = gas_pz / gas_mass + boost_val;
                gas_vx_map[ gas_vx_map == boost_val ] = 0
                gas_vy_map[ gas_vy_map == boost_val ] = 0
                gas_vz_map[ gas_vz_map == boost_val ] = 0

            if pressure_map or temperature_map or metallicity_map:
                t_main = units.gas_code_to_temperature( u_gas , ne_gas )					# K
                rho_main = rho_main * 1e10 * (1.989e33/3.086e21) / (1.6726e-24*3.086e21) / (3.086e21)	# cm^-3


                gas_mass,gas_pressure_map,gas_temperature_map,gas_metallicity_map = rayproj.raytrace_projection_compute( \
                        x[ok], y[ok], z[ok],    \
                        h_main[ok], m_main[ok],   \
                        t_main[ok]*rho_main[ok]*m_main[ok], t_main[ok]*m_main[ok], zm_main[ok]*m_main[ok], \
                        0.0, 0.0, 0.0,          \
                        xrange=xr, yrange=yr, zrange=zr, pixels=pixels )

                gas_pressure_map = gas_pressure_map / gas_mass
                gas_pressure_map[ np.isnan( gas_pressure_map ) ] = 0.0

                gas_temperature_map = gas_temperature_map / gas_mass
                gas_temperature_map[ np.isnan( gas_temperature_map ) ] = 0.0

                gas_metallicity_map =          gas_metallicity_map / gas_mass
                gas_metallicity_map[ np.isnan( gas_metallicity_map ) ] = 0.0

#                gas_pressure,dummy,dummy,dummy = rayproj.gas_raytrace_temperature( \
#                    gas_map_temperature_cuts, \
#                    x[ok], y[ok], z[ok], color_weights[ok], t_main[ok]*rho_main[ok]*m_main[ok], h_main[ok]*2.0, \
#                    xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
#                    isosurfaces = 1, kernel_width=kernel_widths, \
#                    add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]));
#
#                gas_pressure = gas_pressure / out_gas
#                gas_pressure[ np.isnan( gas_pressure ) ] = 0.0

            if temperature_map:
                t_main = units.gas_code_to_temperature( u_gas , ne_gas )
                gas_temperature,dummy,dummy,dummy = rayproj.gas_raytrace_temperature( \
                    gas_map_temperature_cuts, \
                    x[ok], y[ok], z[ok], color_weights[ok],  t_main[ok]*m_main[ok], h_main[ok]*2.0, \
                    xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
                    isosurfaces = 1, kernel_width=kernel_widths, \
                    add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]));

                gas_temperature = gas_temperature / out_gas
                gas_temperature[ np.isnan( gas_temperature ) ] = 0.0

            if density_map:
                gas_density,dummy,dummy,dummy = rayproj.gas_raytrace_temperature( \
                    gas_map_temperature_cuts, \
                    x[ok], y[ok], z[ok], color_weights[ok],  rho_main[ok]*m_main[ok] * 1e10, h_main[ok]*2.0, \
                    xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
                    isosurfaces = 1, kernel_width=kernel_widths, \
                    add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]));

                gas_density = gas_density / out_gas
                gas_density[ np.isnan( gas_density ) ] = 0.0

            if speed_map:
                speed_main = np.sqrt( vx**2 + vy**2 + vz**2 )
                gas_speed,dummy,dummy,dummy = rayproj.gas_raytrace_temperature( \
                    gas_map_temperature_cuts, \
                    x[ok], y[ok], z[ok], color_weights[ok],  speed_main[ok]*m_main[ok], h_main[ok]*2.0, \
                    xrange=xr, yrange=yr, zrange=zr, pixels=pixels, \
                    isosurfaces = 1, kernel_width=kernel_widths, \
                    add_temperature_weights = 0 , KAPPA_UNITS = 2.0885*np.array([1.1,2.0,1.5]));

                gas_speed = gas_speed / out_gas
                gas_speed[ np.isnan( gas_speed ) ] = 0.0

            if np.sum(out_gas)==-1:             # there is nothing here.  zero out images and massmap.
                image24 = np.zeros( (pixels,pixels, 4) )
                image24[:,:,3] = 1.0
                massmap = np.zeros( (pixels,pixels) )
            else:                               # use nice threeeband image bandmaps to convert data to image.
                print("maxden = {:f}  dynrange = {:f}".format( maxden, dynrange ) )
                image24, massmap = \
                    makethreepic.make_threeband_image_process_bandmaps( out_r,out_g,out_u, \
                    maxden=maxden,dynrange=dynrange,pixels=pixels, \
                    color_scheme_nasa=nasa_colors,color_scheme_sdss=sdss_colors);

                image24 = makethreepic.layer_band_images(image24, massmap);


    else:   ## threecolor==0    ( not a three-color image )
        if (do_with_colors==1):
            ## here we have a simple two-pass image where surface brightness is
            ##   luminosity and a 'temperature' is color-encoded
            if (log_temp_wt==1):
                color_weights=np.log10(color_weights); temp_min=np.log10(temp_min); temp_max=np.log10(temp_max);

            massmap_singlelayer,pic_singlelayer,massmap,image24 = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=weights[ok],hsml=h_main[ok],\
                xrange=xr,yrange=yr,
                set_dynrng=dynrange,set_maxden=maxden,
                set_percent_maxden=set_percent_maxden,set_percent_minden=set_percent_minden,
                color_temperature=1,temp_weights=color_weights[ok],
                pixels=pixels,invert_colorscale=invert_colors,
                set_temp_max=temp_max,set_temp_min=temp_min);

        else: ## single color-scale image
            ##
            ## and here, a single-color scale image for projected density of the quantity
            ##
            massmap,image_singledepth = \
            cmakepic.simple_makepic(x[ok],y[ok],weights=weights[ok],hsml=h_main[ok],\
                xrange=xr,yrange=yr,
                set_dynrng=dynrange,set_maxden=maxden,
                set_percent_maxden=set_percent_maxden,set_percent_minden=set_percent_minden,
                color_temperature=0,pixels=pixels,invert_colorscale=1-invert_colors);

            ## convert to actual image using a colortable:
            my_cmap=matplotlib.cm.get_cmap('hot');
            image24 = my_cmap(image_singledepth/255.);

    ##
    ## whichever sub-routine you went to, you should now have a massmap (or set of them)
    ##   and processed rgb 'image24' (which should always be re-makable from the massmaps)
    ##
    ## optionally can have great fun with some lighting effects to give depth:
    ##   (looks great, with some tuning)
    ##
    print("include lighting is set to %d".format(include_lighting) )
    if (include_lighting==1 and np.sum(image24[:,:,:2]) > 0 ):
        #light = matplotlib.colors.LightSource(azdeg=0,altdeg=65)
        print("Lighting is being included!!! " )
        light = viscolors.CustomLightSource(azdeg=0,altdeg=65)
        if (len(massmap.shape)>2):
            ## do some clipping to regulate the lighting:
            elevation = massmap.sum(axis=2)
            minden = maxden / dynrange

            print(" ")
            print(elevation.min(), elevation.max(), elevation.mean() )
            print(minden, maxden )
            print(" " )
            elevation = (elevation - minden) / (maxden - minden)
            elevation[elevation < 0.] = 0.
            elevation[elevation > 1.] = 1.
            elevation *= maxden
            grad_max = maxden / 5.
            grad_max = maxden / 6.
            #image24_lit = light.shade_rgb(image24, massmap.sum(axis=2))
            image24_lit = light.shade_rgb(image24, elevation, vmin=-grad_max, vmax=grad_max)
        else:
            image24_lit = light.shade(massmap, matplotlib.cm.get_cmap('hot'))	#reversed args		# ptorrey -- important
        image24 = image24_lit
    else:
        print("Lighting is not being included :( ")

    plt.imshow(image24,origin='lower',interpolation='bicubic',aspect='auto', extent=[xr[0], xr[1], xr[0], xr[1]]);	# ptorrey normal->auto

    if ((add_gas_layer_to_image=='')==False):
        viscolors.load_my_custom_color_tables();
        plt.imshow(image_singledepth_extra_layer,origin='lower',interpolation='bicubic',aspect='auto',	# ptorrey normal->auto
            cmap=set_added_layer_ctable,alpha=set_added_layer_alpha)

    ## slap on the figure labels/scale bars
    if(show_scale_label==1): overlay_scale_label(xr_0,yr_0,ax_fig,c0='w')
    if(show_time_label==1): overlay_time_label(time,\
        ax_fig,cosmo=cosmo,c0='w',n_sig_add=0,tunit_suffix='Gyr')

    filename='./dummy'
    if(filename_set_manually!=''): filename=filename_set_manually
    #plt.savefig(filename+format,dpi=pixels)#,bbox_inches='tight',pad_inches=0)
    plt.close('all')

    ## save them:
    if False:
        outfi = h5py.File(filename+'.dat','w')
        dset_im = outfi.create_dataset('image24',data=image24)
        dset_mm = outfi.create_dataset('massmap',data=massmap)
        dset_xr = outfi.create_dataset('xrange',data=xr_0)
        dset_yr = outfi.create_dataset('yrange',data=yr_0)
        dset_time = outfi.create_dataset('time',data=time)
        dset_cosmo = outfi.create_dataset('cosmological',data=cosmo)
        if ((add_gas_layer_to_image=='')==False):
            dset_im2 = outfi.create_dataset('image24_extralayer',data=image_singledepth_extra_layer)
        else:
            dset_im2 = outfi.create_dataset('image24_extralayer',data=np.array([-1]))
        outfi.close()

    ## and return the arrays to the parent routine!
    if velocity_map:
        return image24, massmap, gas_vx_map, gas_vy_map, gas_vz_map;
    elif pressure_map:
        return image24, massmap, gas_pressure
    elif temperature_map:
        return image24, massmap, gas_temperature
    elif density_map:
        return image24, massmap, gas_density
    elif speed_map:
        return image24, massmap, gas_speed
    else:
        return image24, massmap;


