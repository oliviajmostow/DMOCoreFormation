
import numpy as np
import matplotlib.pyplot as plt
from readData.DataLoader import DataLoader
from matplotlib.colors import LogNorm
import shared_data
import calculations
import cd_function

rmin = 0.093
rmax = .186
snap = 127
sphere_samples = 200
radial_samples = 30
DesNgb = 32

s8 = ['../updated_m10_713s/s8b1/output','../updated_m10_713s/s8b5/output','../updated_m10_713s/s8b10/output','../updated_m10_713s/s8b20/output','../updated_m10_713s/s8b31/output']

s9 = ['../updated_m10_713s/s9b1/output','../updated_m10_713s/s9b5/output','../updated_m10_713s/s9b10/output','../updated_m10_713s/s9b20/output','../updated_m10_713s/s9b31/output']

s10 = ['../updated_m10_713s/s10b1/output','../updated_m10_713s/s10b5/output','../updated_m10_713s/s10b10/output','../updated_m10_713s/s10b20/output','../updated_m10_713s/s10b31/output']

s7 = ['../updated_m10_713s/s7b1/output','../updated_m10_713s/s7b5/output','../updated_m10_713s/s7b10/output','../updated_m10_713s/s7b20/output','../updated_m10_713s/s7b31/output']

fm = ['../updated_125e8out/b62/output','../updated_125e8out/b31/output','../updated_125e8out/b20/output','../updated_125e8out/b10/output','../updated_m10_713s/s8b5/output','../updated_125e8out/b3/output','../updated_125e8out/b2/output','../updated_125e8out/b1_1/output','../updated_125e8out/b1_2/output','../updated_125e8out/b1_3/output','../updated_125e8out/b1_4/output']

paths=['../m8_dmo_g2/dmo_mass8-output', '../m8_sg_g2_038/output']

def calc_alpha(path):
    cat = DataLoader(path, snap, 1, ['Masses','Coordinates','SubhaloPos'], sub_idx=0)
    coords = cat['PartType1/Coordinates']/0.6909 - cat['SubhaloPos']/0.6909
    masses = cat['PartType1/Masses']*1e10/0.6909
    all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
    all_density = np.empty_like(all_r)
    for i in range(len(all_r)):
        points = calculations.fibonacci_sphere(sphere_samples, all_r[i])
        density = calculations.calc_density(coords, masses, points, DesNgb)
        all_density[i] = (density)
    logr = np.log10(all_r)
    logp = np.log10(all_density)
    h = (logr[1:] - logr[:-1])
    delta_p = logp[1:] - logp[:-1]
    fd = delta_p/h
    #print(fd[4:17])
    #print((0.5*(all_r[1:] + all_r[:-1])[4:17]))
    return np.mean(fd)

def main():
    f = open("m8dmo.txt", "w")
    for i in range(len(paths)):
        a = calc_alpha(paths[i])
        print(a)
        f.write('\n' + str(a))
        #f.write(str(a[1]))
    f.close()
    return


if __name__=='__main__':
	main()
