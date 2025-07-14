import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import cd_function
import shared_data
from readData.DataLoader import DataLoader
import calculations

#parameters
h = 0.6909
a = 0.16
G = 6.67e-11
H = 2.13e-18
fm = ['../updated_125e8out/b62/output','../updated_125e8out/b31/output','../updated_125e8out/b20/output','../updated_125e8out/b10/output','../updated_m10_713s/s8b5/output','../updated_125e8out/b3/output','../updated_125e8out/b2/output','../updated_125e8out/b1_1/output','../updated_125e8out/b1_2/output','../updated_125e8out/b1_3/output','../updated_125e8out/b1_4/output']

path_list = fm
rmin = 0.038
rmax = 10
snap = 127
sphere_samples = 200
radial_samples = 50
DesNgb = 32

#calculate density as a function of radius given params
def core_einasto(log_rho_s, r_s, r_c, radius):
    a = 0.16
    rho_s = 10**log_rho_s
    log_rhoc_rhos = -(2/a) * (np.power((radius + r_c)/r_s, a)-1)
    rhoc_rhos = 10**log_rhoc_rhos
    rho_c = rhoc_rhos * rho_s
    return rho_c

#minimize q statistic wrt all 3 parameters
def core_einasto_fit(r, rho_true, model, guess):
    def merit_func(params):
        #catch for negative values of parameters
        for i in range(3):
            if params[i] <= 0:
                return 1e3
            #if params[2] >= rmax:
                #return 1e9
        rho_hat = model(*params, r)
        N = len(r)
        q2 = 1/N * np.sum(np.square(np.log10(rho_hat) - np.log10(rho_true)))
        return np.sqrt(q2)
    res = scipy.optimize.minimize(merit_func, guess)
    print(merit_func(res.x))
    return res.x

def virial_radius(m):
    virial_radius = np.power(G*m/(100*np.square(H)), 1/3)
    return virial_radius/(3e19)

def alpha(params):
    #average dlogp/dlogr at 1.5% of virial radius
    m = 9.36e9*2e30
    rmin = 0.01* virial_radius(m)
    rmax = 0.02* virial_radius(m)
    r_arr = np.linspace(rmin, rmax, 5)
    rho_s, r_s, r_c = [params[0], params[1], params[2]]
    c = (1 + r_c/r_arr)**((a - 1))
    alpha = -2 * np.power((r_arr/r_s), a) * c
    return np.mean(alpha)


def main():
    f = open("smoothtest.txt", "w")
    for i in range(len(path_list)):
        cat = DataLoader(path_list[i], snap, 1, ['Masses','Coordinates','GroupPos','SubhaloMass','SubhaloPos', 'GroupMassType'], sub_idx=0)
        coords = cat['PartType1/Coordinates']/h - cat['SubhaloPos']/h
        masses = cat['PartType1/Masses']*1e10/h
    #calculate true density
        rho_true = []
        all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
        for r in all_r:
            points = calculations.fibonacci_sphere(sphere_samples, r)
            density = calculations.calc_density(coords, masses, points, DesNgb)
            rho_true.append(density)
        rho_true = np.array(rho_true)
        fitted_params = core_einasto_fit(all_r, rho_true, core_einasto, [5.5, 2, .1])
        print('best fit parameters rho_s, r_s, r_c: ' + str(fitted_params))
        print('Rvir: ', virial_radius(7.4e7*2e30))
        #print('alpha: ' + str(alpha(fitted_params)) + ' calculated at: ' + str(0.015*virial_radius(9e9*2e30)))
        f.write('\n' + str(fitted_params[2]))
    f.close()
    #print('alpha: ' + str(alpha(fitted_params)) + ' calculated at: ' + str(0.015*virial_radius(9e9*2e30)))
    
    #plot model over density profile
    fig, ax = shared_data.set_plot_params()
    ax.plot(all_r, rho_true, color='black', label='data')
    ax.plot(all_r, core_einasto(*fitted_params, all_r), color='orchid', linestyle='dashed', label='model fit')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.axvline(x=0.038)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Density ($M_{\odot}$/kpc)')
    fig.savefig('core_ein_fit.pdf', bbox_inches='tight')
    return

if __name__=="__main__":
    main()

