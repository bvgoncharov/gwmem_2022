import numpy as np
import pandas as pd
import requests

from astropy.cosmology import Planck18, z_at_value
import astropy.units as uu

out_name = '../pops_gwmem_2022/GWTC-like_population_20230314.hdf5'

gwosc_parameters = {}

rr = requests.get(
  "https://www.gwosc.org/eventapi/jsonfull/query/show",
  params=gwosc_parameters,
  ).json()

columns = ['ra', 'dec', 'psi', 'phase', 'luminosity_distance', 'redshift', 'mass_1', 'mass_2', 'geocent_time', 'theta_jn', 'a_1', 'a_2', 'version']

events = pd.DataFrame(columns=columns)

for key, entry in rr['events'].items():
  if key[0:2] != 'GW':
    # Skipping marginal events
    continue

  name = entry['commonName']

  # Masses
  mass_1 = entry['mass_1_source']
  mass_2 = entry['mass_2_source']
  if mass_1 is None:
    continue
  qq = min((mass_1, mass_2)) / max((mass_1, mass_2))

  # Sky position, polarization, phase, spins, inclination: random
  ra = np.random.uniform(0, 2*np.pi)
  dec = np.arcsin(np.random.uniform(-1, 1))
  psi = np.random.uniform(0, 2*np.pi)
  phase = np.random.uniform(0, 2*np.pi)
  theta_jn = np.arcsin(np.random.uniform(-1, 1))
  a_1 = (1 + qq) * entry['chi_eff']
  a_2 = 0.

  # Distance and redshift
  if entry['luminosity_distance_unit'] != 'Mpc':
    print('[!] Luminosity distance found, ', entry['luminosity_distance'], '. Unit is ', entry['luminosity_distance_unit'], ' (not Mpc).')
  luminosity_distance = entry['luminosity_distance']
  redshift = z_at_value(Planck18.luminosity_distance, luminosity_distance * uu.Mpc).value

  geocent_time = entry['GPS']

  version = entry['version']

  if entry['commonName'] in events.index:
    if version <= events['version'][name]:
      continue

  events.loc[name] = [ra, dec, psi, phase, luminosity_distance, redshift, mass_1, mass_2, geocent_time, theta_jn, a_1, a_2, version]

events = events.drop(columns=['version'])

events.to_hdf(out_name, mode='w', key='root')
