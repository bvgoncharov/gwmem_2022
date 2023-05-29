import numpy as np
import pandas as pd

from astropy.cosmology import Planck18, z_at_value
import astropy.units as uu

out_name = '/home/bgonchar/pops_gwmem_2022/LISA_test_BBH_20230511.hdf5'

columns = ['ra', 'dec', 'psi', 'phase', 'luminosity_distance', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'theta_jn', 'a_1', 'a_2']

events = pd.DataFrame(columns=columns)

ii = 0
for mass in [10**6, 5*10**6, 10**7]: # M_Solar
  for dist in [1000,2000,5000,10000]: # Mpc

    ii+=1
    name = str(ii)
  
    # Masses
    mass_1 = 1.001 * mass
    mass_2 = mass
    if mass_1 is None:
      continue
    qq = min((mass_1, mass_2)) / max((mass_1, mass_2))
  
    # Sky position, polarization, phase, spins, inclination: random
    ra = np.random.uniform(0, 2*np.pi)
    dec = np.arcsin(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, 2*np.pi)
    phase = np.random.uniform(0, 2*np.pi)
    theta_jn = np.arcsin(np.random.uniform(-1, 1))
    a_1 = 0.
    a_2 = 0.
  
    luminosity_distance = dist
    redshift = z_at_value(Planck18.luminosity_distance, dist * uu.Mpc).value
  
    geocent_time = 0.
  
    events.loc[name] = [ra, dec, psi, phase, luminosity_distance, redshift, mass_1, mass_2, geocent_time, theta_jn, a_1, a_2]

events.to_hdf(out_name, mode='w', key='root')
