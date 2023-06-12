import numpy as np
import pandas as pd

from astropy.cosmology import Planck18, z_at_value
import astropy.units as uu

out_name = '/home/bgonchar/pops_gwmem_2022/test_gw150914like_theta_20230606.hdf5'

columns = ['ra', 'dec', 'psi', 'phase', 'luminosity_distance', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'theta_jn', 'a_1', 'a_2']

events = pd.DataFrame(columns=columns)

mass_1 = 1.001 * 30.0
mass_2 = 30.0
qq = min((mass_1, mass_2)) / max((mass_1, mass_2))
# Sky position, polarization, phase, spins, inclination: random
ra = np.random.uniform(0, 2*np.pi)
dec = np.arcsin(np.random.uniform(-1, 1))
psi = np.random.uniform(0, 2*np.pi)
phase = np.random.uniform(0, 2*np.pi)
a_1 = 0.
a_2 = 0.
dist = 300
luminosity_distance = dist
redshift = z_at_value(Planck18.luminosity_distance, dist * uu.Mpc).value
geocent_time = 0.

ii = 0
for theta_jn in [-np.pi/2, 0., np.pi/4, np.pi/3, np.pi/2]:

    ii+=1
    name = str(ii)
  
    events.loc[name] = [ra, dec, psi, phase, luminosity_distance, redshift, mass_1, mass_2, geocent_time, theta_jn, a_1, a_2]

events.to_hdf(out_name, mode='w', key='root')
