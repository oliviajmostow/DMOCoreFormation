import numpy as np


def determine_image_stretch( data, dynrange=None, maxden=None, set_dynrng=None, set_maxden=None, **kwargs):
#    print dynrng, maxden, set_dynrng, set_maxden
    if( (dynrange != None) and (maxden != None) ):
        data.set_dynrng = dynrange
        data.set_maxden = maxden
    elif( (set_dynrng != None) and (set_maxden != None) ):
        data.set_dynrng = set_dynrng
        data.set_maxden = set_maxden
    else:
        print("NO DYNRNG/MAXDEN INFO FOUND!  SETTING TO DEFAULT VAUES!" )
        data.set_dynrng = 1e3
        data.set_maxden = 1.0

    return data

def determine_image_bounds( data, xrange=None, yrange=None, zrange=None, **kwargs):
    if ( xrange is not None):               # assume here that we use manually prescribed values
        if (yrange is None): yrange= xrange         # if yrange is set, use that value.  Otherwise, copy xrange
        if (zrange is None): zrange = xrange        # same for zr
    else:
        print("No bounds detected [xr,yr,zr] in 'utilities.determine_image_bounds'.  Using min/max values" )
        x_stretch = np.max( data.pos[:,0]) - np.min(data.pos[:,0])
        y_stretch = np.max( data.pos[:,1]) - np.min(data.pos[:,1])
        x_mid     = 0.5*( np.max( data.pos[:,0]) + np.min(data.pos[:,0]) )
        y_mid     = 0.5*( np.max( data.pos[:,1]) + np.min(data.pos[:,1]) )
        z_mid     = 0.5*( np.max( data.pos[:,2]) + np.min(data.pos[:,2]) )

        stretch = np.max( [x_stretch, y_stretch] )

        xrange = [ x_mid - stretch/2.0 , x_mid + stretch/2.0 ]
        yrange = [ y_mid - stretch/2.0 , y_mid + stretch/2.0 ]
        zrange = [ z_mid - stretch/2.0 , z_mid + stretch/2.0 ]


    data.xr = xrange
    data.yr = yrange
    data.zr = zrange
    return data


def set_band_ids( data, BAND_IDS=[9,10,11], **kwargs):
    data.BAND_IDS = BAND_IDS
    return data

def set_kappa_units( data, kappa_units=2.08854068444, kpc=True, mpc=False, kappa_factor=1.0, **kwargs):
    if kpc:
        if mpc:
            print("ERROR: It appears (in image_properties) that the code thinks both kpc and mpc are set to true.  This cannot be true, and is messing up how the kappa_units (opacity) is set" )
            sys.exit()
        data.kappa_units = kappa_units
    else:
        data.kappa_units = kappa_units * 1e-6 
    if kappa_factor != 1.0:
        print("Kappa factor is not unity.  Adjusting accordingly..." )
        data.kappa_units *= kappa_factor 

    return data

def set_imf( data, IMF_SALPETER=0, IMF_CHABRIER=1, **kwargs):
    if IMF_SALPETER:
        if IMF_CHABRIER:
            print("ERROR:  It sppears (in image properties) that the code thinks the IMF is set to both chabrier and salpeter, and this is messing up how the IMF is set.")
        data.imf_salpeter = 1
        data.imf_chabrier = 0
    else:
        data.imf_salpeter = 0
        data.imf_chabrier = 1
    return data


def set_dust_to_gas_ratio( data, dust_to_gas_ratio_rescale = 1.0, **kwargs):
    data.dust_to_gas_ratio_rescale = dust_to_gas_ratio_rescale
    return data
        
