
import numpy as np
from readData.DataLoader import DataLoader
from scipy import stats
import h5py, os
from scipy.spatial import KDTree

#parameters
snap = 127
part_type = 1
h = 0.6909
rmin = .1
rmax = 30
sphere_samples = 200
radial_samples = 50
DesNgb = 32


#equally spaced pts on sphere
def fibonacci_sphere(samples, r):   
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / (samples - 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points) * r

    return points


def calc_density(matter_coords, matter_mass, sample_coords, DesNgb):
    tree = KDTree(matter_coords)
    distance, idx = tree.query(sample_coords, DesNgb)
    hsml = distance[:,-1]
    mass_enclosed = np.sum(matter_mass[idx], axis=1)
    density =  mass_enclosed / (4 / 3 * np.pi * np.power(hsml,3))
    density = np.average(density)
    return density


