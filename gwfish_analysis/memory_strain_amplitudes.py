"""
Accompanying code for gwfish_memory_pipeline.py
Determines maximum strain amplitude for J_E and J_J for a given injection
"""

import os
import json
import time

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
max_je_jj = np.zeros( ((opts.inj+1)*opts.num-opts.inj*opts.num,3) ) # 2 max strain keys + three snrs (snr_je, jj, jejj) for all detectors
#je_pol = np.zeros((2, 129793))
#jj_pol = np.zeros((2, 129793))
#jejj_pol = np.zeros((2, 129793))
jejj_pol_holder = np.zeros((2, 129793))

out_file_name = totaldir+'max_je_jj_ET_'+str(opts.num)+'_'+str(opts.inj)+'.txt'
if os.path.exists(out_file_name):
  max_je_jj = np.empty((0,3))
  for ii in range(100):
    out_file_name = totaldir+'max_je_jj_ET_'+str(opts.num)+'_'+str(ii)+'.txt'
    if os.path.exists(out_file_name):
      max_je_jj = np.vstack([max_je_jj,np.loadtxt(out_file_name)])
    else:
      print('File not found, ', out_file_name)
  par_subset = parameters[0:10000]

  max_je_jj_pd = pd.DataFrame(max_je_jj, index=par_subset.index, columns=['snr_je', 'snr_jj', 'snr_jejj'])
  par_subset = pd.concat([par_subset,max_je_jj_pd],axis=1)  
  par_subset['mass_ratio'] = par_subset['mass_1']/par_subset['mass_2']

  # Let us show that sorting by SNR J_J would also be effective for SNR J_E
  # It appears less effective, but more effective than sorting by distance
  idxs = np.arange(1,10001,1)
  # J_J
  snr_jj_accumulated = np.sqrt(np.cumsum(par_subset.sort_values('snr_jj',ascending=False)['snr_jj']**2))
  plt.plot(idxs,snr_jj_accumulated,label='SNR JJ, sorted by SNR JJ')
  plt.plot(idxs,np.sqrt(np.cumsum(par_subset.sort_values('snr_je',ascending=False)['snr_jj']**2)),label='SNR JJ, sorted by SNR JE')
  plt.plot(idxs,np.sqrt(np.cumsum(par_subset.sort_values('luminosity_distance',ascending=True)['snr_jj']**2)),label='SNR JJ, sorted by distance')
  snr_jj_tot = max(snr_jj_accumulated)
  dist_const = max(np.sqrt(np.cumsum(1/par_subset.sort_values('luminosity_distance',ascending=True)['luminosity_distance']**2)))
  plt.plot(idxs,snr_jj_tot/dist_const*np.sqrt(np.cumsum(1/par_subset.sort_values('luminosity_distance',ascending=True)['luminosity_distance']**2)),label='Analytical dist. dependence')
  plt.xlabel('Event number')
  plt.ylabel('Cumulative SNR')
  plt.legend()
  plt.savefig('/home/bgonchar/out_gwmem_2022/sorting_snr_jj.png')
  plt.close()
  # J_J log-log
  plt.loglog(idxs,snr_jj_accumulated,label='SNR JJ, sorted by SNR JJ')
  plt.loglog(idxs,np.sqrt(np.cumsum(par_subset.sort_values('snr_je',ascending=False)['snr_jj']**2)),label='SNR JJ, sorted by SNR JE')
  plt.loglog(idxs,np.sqrt(np.cumsum(par_subset.sort_values('luminosity_distance',ascending=True)['snr_jj']**2)),label='SNR JJ, sorted by distance')
  plt.xlabel('Event number')
  plt.ylabel('Cumulative SNR')
  plt.legend()
  plt.savefig('/home/bgonchar/out_gwmem_2022/sorting_snr_jj_loglog.png')
  plt.close()
  # J_E
  plt.plot(idxs,np.sqrt(np.cumsum(par_subset.sort_values('snr_jj',ascending=False)['snr_je']**2)),label='SNR JE, sorted by SNR JJ')
  plt.plot(idxs,np.sqrt(np.cumsum(par_subset.sort_values('snr_je',ascending=False)['snr_je']**2)),label='SNR JE, sorted by SNR JE')
  plt.plot(idxs,np.sqrt(np.cumsum(par_subset.sort_values('luminosity_distance',ascending=True)['snr_je']**2)),label='SNR JE, sorted by distance')
  plt.xlabel('Event number')
  plt.ylabel('Cumulative SNR')
  plt.legend()
  plt.savefig('/home/bgonchar/out_gwmem_2022/sorting_snr_je.png')
  plt.close()
  # SNR against distance
  plt.loglog(par_subset['luminosity_distance'], par_subset['snr_jj'])
  dist_inv_prop = par_subset['snr_jj'].iloc[0]*par_subset['luminosity_distance'].iloc[0]/par_subset['luminosity_distance']
  plt.loglog(par_subset['luminosity_distance'], dist_inv_prop)
  plt.xlabel('Luminosity distance [Mpc]')
  plt.ylabel('SNR JJ')
  plt.savefig('/home/bgonchar/out_gwmem_2022/dist_snr_jj.png')
  plt.close()
  plt.loglog(par_subset['luminosity_distance'], par_subset['snr_je'])
  plt.xlabel('Luminosity distance [Mpc]')
  plt.ylabel('SNR JE')
  plt.savefig('/home/bgonchar/out_gwmem_2022/dist_snr_je.png')
  plt.close()

  ## Saving sub-populations sorted by SNR J_J and SNR J_E
  par_subset_jjs = par_subset.sort_values('snr_jj',ascending=False)
  par_subset_jjs.to_hdf(opts.injfile.replace('.hdf5','_sorted_snrjj.hdf5'), mode='w', key='root')
  par_subset_jes = par_subset.sort_values('snr_je',ascending=False)
  par_subset_jes.to_hdf(opts.injfile.replace('.hdf5','_sorted_snrje.hdf5'), mode='w', key='root')

  # In case corner plot is made
  del par_subset['J_E']
  del par_subset['J_J']
  del par_subset['a_1']
  del par_subset['a_2']
else:
  time_start = time.time()
  for kk in range(opts.inj*opts.num, (opts.inj+1)*opts.num):

    outfile_name = namebase+'_'+str(kk)+'.pkl'

    parameter_values = parameters.iloc[kk]
    print(parameter_values)
  
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
                  }

    waveform_obj = waveform_class(opts.waveform, parameter_values, data_params)
    try:
      _ = waveform_obj()
    
      #max_je_jj[opts.inj*opts.num+kk,0] = np.max(np.abs(waveform_obj.J_E[0] -1j * waveform_obj.J_E[1])) # max_je
      #max_je_jj[opts.inj*opts.num+kk,1] = np.max(np.abs(waveform_obj.J_J[0] -1j * waveform_obj.J_J[1])) # max_jj
    
      # Memory SNR calculation
      #je_pol = np.vstack([waveform_obj.J_E_tilde[0],waveform_obj.J_E_tilde[1]])
      #jj_pol = np.vstack([waveform_obj.J_J_tilde[0],waveform_obj.J_J_tilde[1]])
      #jejj_pol = np.vstack([waveform_obj.J_E_tilde[0]+waveform_obj.J_J_tilde[0],waveform_obj.J_E_tilde[1]+waveform_obj.J_J_tilde[1]])

      # Let us try to break this for memory optimization
      jejj_pol_holder = np.vstack([waveform_obj.J_E_tilde[0],waveform_obj.J_E_tilde[1]])
      max_je_jj[kk-opts.inj*opts.num,0] = calculate_snr_mem(network, parameter_values, jejj_pol_holder, waveform_obj.t_of_f)

      jejj_pol_holder = np.vstack([waveform_obj.J_J_tilde[0],waveform_obj.J_J_tilde[1]])
      max_je_jj[kk-opts.inj*opts.num,1] = calculate_snr_mem(network, parameter_values, jejj_pol_holder, waveform_obj.t_of_f)

      jejj_pol_holder = np.vstack([waveform_obj.J_E_tilde[0]+waveform_obj.J_J_tilde[0],waveform_obj.J_E_tilde[1]+waveform_obj.J_J_tilde[1]])
      max_je_jj[kk-opts.inj*opts.num,2] = calculate_snr_mem(network, parameter_values, jejj_pol_holder, waveform_obj.t_of_f)

      # In case necessary to save polarizations for plotting (publfig_mem_demo.py)
      #mem_pol = np.vstack([waveform_obj.frequencyvector,np.abs(waveform_obj.frequency_domain_strain[:,0] - 1j*waveform_obj.frequency_domain_strain[:,1]),np.abs(waveform_obj.J_E_tilde[0]-1j*waveform_obj.J_E_tilde[1]),np.abs(waveform_obj.J_J_tilde[0]-1j*waveform_obj.J_J_tilde[1]),np.abs(waveform_obj.J_E_tilde[0]+waveform_obj.J_J_tilde[0] - 1j*(waveform_obj.J_E_tilde[1]+waveform_obj.J_J_tilde[1]))]).T
      #np.savetxt(totaldir + outfile_name.replace('.pkl','_fd_waveform_mem.txt'), mem_pol)
      #mem_pol_td = np.vstack([waveform_obj.timevector, waveform_obj._lal_ht_plus.data.data, waveform_obj._lal_ht_cross.data.data, waveform_obj.J_E[0], waveform_obj.J_E[1], waveform_obj.J_J[0], waveform_obj.J_J[1]]).T
      #np.savetxt(totaldir + outfile_name.replace('.pkl','_td_waveform_mem.txt'), mem_pol_td)

      #for ii, det in enumerate(det_keys):
      #network = gw.detection.Network([det], detection_SNR=threshold_SNR, parameters=parameters, fisher_parameters=fisher_parameters, config=opts.config)
      #je_proj = gw.detection.projection(parameter_values, network.detectors[0], je_pol.T, waveform_obj.t_of_f)
      #jj_proj = gw.detection.projection(parameter_values, network.detectors[0], jj_pol.T, waveform_obj.t_of_f)
      #jejj_proj = gw.detection.projection(parameter_values, network.detectors[0], jejj_pol.T, waveform_obj.t_of_f)

      # After first two columns being max_je/jj,
      # Next n_det columns are snr_je, next n_det columns snr_jj, and final
      # n_det columns are for snr_je_jj
      #max_je_jj[opts.inj*opts.num+kk,ii+2+len(det_keys)*0] = np.sqrt(np.sum(gw.detection.SNR(network.detectors[0], je_proj)**2))
      #max_je_jj[opts.inj*opts.num+kk,ii+2+len(det_keys)*1] = np.sqrt(np.sum(gw.detection.SNR(network.detectors[0], jj_proj)**2))
      #max_je_jj[opts.inj*opts.num+kk,ii+2+len(det_keys)*2] = np.sqrt(np.sum(gw.detection.SNR(network.detectors[0], jejj_proj)**2))

      # Deleting this actually seems to matter for RAM usage
      del waveform_obj
    except:
      print('Error in ', kk)
      #max_je_jj[opts.inj*opts.num+kk,:] += np.array([[0. for ii in range(20)]])
      continue
  
  np.savetxt(totaldir+'max_je_jj_'+opts.det+'_'+str(opts.num)+'_'+str(opts.inj)+'.txt', max_je_jj)
  
  print('Completed')
  print('Elapsed time: ', time.time()-time_start)
