"""
This is after gwfish_memory_postprocessing.py
So, just repeat the slurm script structure, replacing python script by this one.

Make sure to have the same --num, so that results are loaded correctly.
"""

import os
import tqdm
import pickle
import json

import lal

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer

import gwfish_utils as gu
import plotting_utils as pu
import GWFish.modules as gw

def extra_model(am, opts):
    """
    There are 4 combinations of J_E and J_J between 0 and 1.
    By choosing one option for J_E/J_J, we can fix one extra
    for the opposite, and get all combinations.

    Input: 
    am: one alternative model, tuple, e.g. ('J_E', 1.0)
    opts: command line options

    Output: alternative model to am
    """
    if am[0]=='J_J' and am[1]==opts.j_j:
      # JJ right, JE wrong
      extra = ('J_E', 1 - opts.j_e)
    elif am[0]=='J_E' and am[1]==opts.j_e:
      # Both terms right
      extra = ('J_J', opts.j_j)
    elif am[0]=='J_J' and am[1]==1-opts.j_j:
      # JJ wrong, JE right
      extra = ('J_E', opts.j_e)
    elif am[0]=='J_E' and am[1]==1-opts.j_e:
      # Both terms wrong
      extra = ('J_J', 1 - opts.j_j)
    else:
      extra = None
    return extra

def model_label(am):
    return am[0].replace('_','')+str(int(am[1]))

outdir = '/home/bgonchar/out_gwmem_2022/publ_fig/'

#all_detectors = ['CE1','ET']
#all_detectors = ['ET','CE1']
all_detectors = ['ET']#,'CE1','LLO','LHO','VIR','VOY']
#detector_combinations = [['ET'],['ET','CE1']]
detector_combinations = [['ET']]#,['ET','CE1'],['LLO','LHO','VIR'],['VOY','VIR']]
#all_detectors = ['LISA']
#detector_combinations = [['LISA']]

opts = gu.parse_commandline()
range_simulations = range(opts.inj*opts.num, (opts.inj+1)*opts.num)

popdir, totaldir, namebase = gu.output_names(opts)

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

alternative_models = [('J_E', opts.j_e), ('J_E', 1 - opts.j_e), ('J_J', opts.j_j), ('J_J', 1 - opts.j_j)]
amod_labels = [model_label(am) for am in alternative_models]
extra_labels = [model_label(am)+model_label(extra_model(am,opts)) for am in alternative_models]
# Accounting for noise realizations
_amod_labels = []
_extra_labels = []
_alternative_models = []
for nr in range(opts.noise):
  _alternative_models += [am for am in alternative_models]
  if nr==0:
    _amod_labels += [am for am in amod_labels]
    _extra_labels += [am for am in extra_labels]
  else:
    _amod_labels += [am+'_'+str(nr) for am in amod_labels]
    _extra_labels += [am+'_'+str(nr) for am in extra_labels]
amod_labels = _amod_labels
extra_labels = _extra_labels
alternative_models = _alternative_models

logz_detcombs = {''.join(dtcomb): pd.DataFrame(columns=amod_labels+extra_labels,index=parameters.index, dtype=np.float64) for dtcomb in detector_combinations}
jejj_covs = {''.join(dtcomb): pd.DataFrame(columns=['JE','JJ','JEJJ','dJE','dJJ'],index=parameters.index, dtype=np.float64) for dtcomb in detector_combinations}

largest_slurm_index = 49

# Combining results
for inj in range(largest_slurm_index+1): #tqdm.tqdm(range(len(parameters))):

  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)

    # Without noise
    #logzs_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(inj)+'_logzs.json'
    # With noise
    logzs_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(inj)+'_noise_'+str(opts.noise)+'_logzs.json'

    if opts.randomize_mem_pe:
      jejj_cov_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(inj)+'_covjejj_noise.json'
    else:
      jejj_cov_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(inj)+'_covjejj.json'

    #logz_temp = pd.read_hdf(logzs_file_name)
    #logz_detcombs[dtcomb_label].loc[logz_temp.index] = logz_temp
    jejj_temp = pd.read_hdf(jejj_cov_file_name)
    jejj_covs[dtcomb_label].loc[jejj_temp.index] = jejj_temp

# Save combined results
for dtcomb in detector_combinations:
  dtcomb_label = ''.join(dtcomb)
  #dict_z = logz_detcombs[dtcomb_label].iloc[0:(largest_slurm_index+1)*opts.num]
  #combined_logzs_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(largest_slurm_index)+'_noise_'+str(opts.noise)+'_logzs.json'
  #dict_z.to_hdf(combined_logzs_file_name, mode='w', key='root')
  #print('Saved:',combined_logzs_file_name)
  dict_covjejj = jejj_covs[dtcomb_label].iloc[0:(largest_slurm_index+1)*opts.num]
  if opts.randomize_mem_pe:
    combined_covjejj_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(largest_slurm_index)+'_covjejj_noise.json'
  else:
    combined_covjejj_file_name = totaldir+namebase+'_'+str((inj+1)*opts.num-1)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(largest_slurm_index)+'_covjejj.json'
  dict_covjejj.to_hdf(combined_covjejj_file_name, mode='w', key='root')
  print('Saved:',combined_covjejj_file_name)

# General and universal outlook
#for dtcomb in detector_combinations:
#  dtcomb_label = ''.join(dtcomb)
#  #dict_z = pd.DataFrame.from_dict(logz_detcombs[dtcomb_label])
#  dict_z = logz_detcombs[dtcomb_label].iloc[0:(largest_slurm_index+1)*opts.num]
#
#  #import ipdb; ipdb.set_trace()
#  je1jj1_ebms_key = 'JE1JJ1' # BMS true: JJ1JE1
#  je0jj0_poincare_key = 'JE0JJ0' # BMS true: JJ0JE0
#  je1jj0_bms_key = 'JJ0JE1' # BMS true: JE1JJ0
#  
#  #je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
#
#  ebms_vs_bms = dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key]
#  ebms_vs_poincare = dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key]
#  bms_vs_poincare = dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key]
#  
#  #spinmonly_vs_poincare = dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key]
#
#  # Uncomment in case we need to sort by geocent time
#  #geocent_time = parameters['geocent_time'].iloc[0:(largest_slurm_index+1)*opts.num]/60/60/24 # units days
#  #geocent_time = geocent_time.sort_values()
#  #ebms_vs_bms = ebms_vs_bms.reindex(geocent_time.index)
#  #ebms_vs_poincare = ebms_vs_poincare.reindex(geocent_time.index)
#  #bms_vs_poincare = bms_vs_poincare.reindex(geocent_time.index)
#  #geocent_time = geocent_time.values
#
#  # Plot sorted evidence for model combinations as a function of event number
#  fig, ax = plt.subplots(figsize=(8, 5))
#  indices = np.linspace(1,len(bms_vs_poincare),len(bms_vs_poincare))
#  indices_major = np.arange(1,len(bms_vs_poincare),15)
#  # Sorting by evidence
#  ax.loglog(indices, bms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{BMS}}_{Poincare}$',color='red')
#  ax.loglog(indices, ebms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{Poincare}$',color='green')
#  ax.loglog(indices, ebms_vs_bms.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{BMS}$',color='blue')
#  # Sorting by time
#  #ax.plot(geocent_time, bms_vs_poincare.cumsum().values,label='$\ln \mathcal{B}^{\mathrm{BMS}}_{Poincare}$',color='red')
#  #ax.plot(geocent_time, ebms_vs_poincare.cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{Poincare}$',color='green')
#  #ax.plot(geocent_time, ebms_vs_bms.cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{BMS}$',color='blue')
#  # Noise realizations
#  #for nn in tqdm.tqdm(range(1,opts.noise)):
#  #  je1jj1_ebms_key = 'JE1JJ1_'+str(nn)
#  #  je0jj0_poincare_key = 'JE0JJ0_'+str(nn)
#  #  je1jj0_bms_key = 'JJ0JE1_'+str(nn)
#  #  ebms_vs_bms_n = dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key]
#  #  ebms_vs_poincare_n = dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key]
#  #  bms_vs_poincare_n = dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key]
#  #  ax.loglog(indices, bms_vs_poincare_n.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{BMS}}_{Poincare}$',color='red',alpha=0.1,lw=0.5)
#  #  ax.loglog(indices, ebms_vs_poincare_n.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{Poincare}$',color='green',alpha=0.1,lw=0.5)
#  #  ax.loglog(indices, ebms_vs_bms_n.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{BMS}$',color='blue',alpha=0.1,lw=0.5)
#
#  ax.axhline(5., color='red', linestyle=':', label='Detection threshold')
#  ax.legend()
#  #ax.set_xticks(indices_major)
#  #ax.set_xticks(indices, minor=True)
#  #ax.grid(which='both')
#  ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$')
#  ax.set_xlabel('Number of detections, sorted by evidence')
#  #plt.xlim([1,indices[-1]])
#  plt.tight_layout()
#  plt.savefig(outdir+'_evidence_n_'+opts.label+'_'+dtcomb_label+'.png')
#  plt.close()

import ipdb; ipdb.set_trace()



