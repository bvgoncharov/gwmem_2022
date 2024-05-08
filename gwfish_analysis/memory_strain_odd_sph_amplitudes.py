"""
Accompanying code for gwfish_memory_pipeline.py
Determines SNR of m-odd spherical harmonic modes of the waveform, which is 
required to break the degeneracy between GW polarization angle and phase. 
The complete degeneracy with (2,2) mode only makes the total memory SNR 
to be added in the powers of 4, not powers of 2, making evidence weaker.
"""

import os
import json
import time
import tqdm

import numpy as np
import pandas as pd

from numpy.random import default_rng

from itertools import combinations, chain

import GWFish.modules as gw

import gwfish_utils as gu

from matplotlib import pyplot as plt

rng = default_rng()

def mismatch(ww1, ww2):
    return 1 - np.vdot(ww1, ww2) / np.vdot(ww1, ww1) ** 0.5 / np.vdot(ww2, ww2) ** 0.5

def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]

def cumulative_snr(snrs):
    return np.sqrt(np.cumsum(snrs**2))

def calculate_snr_mem(network,parameter_values,mem_polarizations,t_of_f):
    mem_proj = gw.detection.projection(parameter_values, network.detectors[0], mem_polarizations.T, waveform_obj.t_of_f)
    return np.sqrt(np.sum(gw.detection.SNR(network.detectors[0], mem_proj)**2))

opts = gu.parse_commandline()

popdir, totaldir, namebase = gu.output_names(opts)
from chainconsumer import ChainConsumer

if not os.path.exists(totaldir):
    print('Output directory does not exist: ', totaldir)
    raise ValueError()

threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

#det_keys = [opts.det] #'ET','CE1','LLO','LHO','VIR','VOY'] # Save on memory?

np.random.seed(0)

waveform_class = eval(opts.waveform_class)

# Reserve memory in advance
snr_odd_sph = np.zeros( ((opts.inj+1)*opts.num-opts.inj*opts.num,1) ) # 2 max strain keys + three snrs (snr_je, jj, jejj) for all detectors
#je_pol = np.zeros((2, 129793))
#jj_pol = np.zeros((2, 129793))
#jejj_pol = np.zeros((2, 129793))
jejj_pol_holder = np.zeros((2, 129793))

out_file_name = totaldir+'snr_sph_odd_'+opts.det+'_'+str(opts.num)+'_'+str(opts.inj)+'.txt'
print(out_file_name)
if os.path.exists(out_file_name):
  pass
else:
  time_start = time.time()
  for kk in tqdm.tqdm(range(opts.inj*opts.num, (opts.inj+1)*opts.num)):

    outfile_name = namebase+'_'+str(kk)+'.pkl'

    parameter_values = parameters.iloc[kk]
    #print(parameter_values)
  
    network = gw.detection.Network([opts.det], detection_SNR=threshold_SNR, parameters=parameters, fisher_parameters=fisher_parameters, config=opts.config)
  
    #for dd in np.arange(len(network.detectors)):
    dd = 0

    cache_option = False
    data_params = {
                      'frequencyvector': network.detectors[dd].frequencyvector,
                      'memory_contributions': opts.mem_sim, #'J_E, J_J', # J_E, J_J
                      'time_domain_f_min': opts.td_fmin,
                      'f_ref': opts.f_ref,
                      'cache_waveforms': cache_option,
                      'only_odd_m': True,
                  }

    waveform_obj = waveform_class(opts.waveform, parameter_values, data_params)
    try:
      _ = waveform_obj()
    
      snr_odd_sph[kk-opts.inj*opts.num,0] = calculate_snr_mem(network, parameter_values, waveform_obj.frequency_domain_strain.T, waveform_obj.t_of_f)

      # Deleting this actually seems to matter for RAM usage
      del waveform_obj
    except:
      print('Error in ', kk)
      #snr_odd_sph[opts.inj*opts.num+kk,:] += np.array([[0. for ii in range(20)]])
      continue
  
  np.savetxt(totaldir+'snr_odd_sph_'+opts.det+'_'+str(opts.num)+'_'+str(opts.inj)+'.txt', snr_odd_sph)
  
  print('Completed')
  print('Elapsed time: ', time.time()-time_start)
