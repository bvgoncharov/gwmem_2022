import os
import tqdm
import pickle
import json

import lal

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

for dtc in all_detectors:
  parameters['SNR'+dtc] = np.zeros(len(parameters))

threshold_SNR = np.array([0., 9.])
if all_detectors[0]=='LISA':
    svd_threshold = 1e-20
else:
    svd_threshold = opts.svd_threshold

no_result = []
logz_detcombs = {''.join(dtcomb): pd.DataFrame(columns=amod_labels+extra_labels,index=parameters.iloc[range_simulations].index, dtype=np.float64) for dtcomb in detector_combinations}
#list_JEJJ_cov_dicts = {''.join(dtcomb): [] for dtcomb in detector_combinations}
if opts.randomize_mem_pe:
    jejj_columns_covs = ['JE','JJ','JEJJ','dJE','dJJ']
else:
    jejj_columns_covs = ['JE','JJ','JEJJ']
jejj_covs = {''.join(dtcomb): pd.DataFrame(columns=jejj_columns_covs,index=parameters.iloc[range_simulations].index, dtype=np.float64) for dtcomb in detector_combinations}

errors_diag = np.zeros((len(parameters), 13))

for kk in tqdm.tqdm(range_simulations): #tqdm.tqdm(range(len(parameters))):
  kk_idx = parameters.index[kk]

  outfile_name = namebase+'_'+str(kk)+'.pkl'
  parameter_values = parameters.iloc[kk]
  if os.path.exists(totaldir + outfile_name):
    with open(totaldir + outfile_name, 'rb') as fh:
      fm_object = pickle.load(fh)
  else:
    no_result.append(kk)
    print('No result ', kk)
    continue


  # Saving the Fisher matrix prior to frequency range cut, for inspection
  fm_old = fm_object.fm
  errors_old, _ = gw.fishermatrix.invertSVD(fm_old, thresh=svd_threshold)

  errors_diag[kk,:] = np.diag(errors_old)

  del fm_object

# Saving
np.savetxt(opts.outdir+popdir+'/'+namebase+'_'+str(opts.num)+'_diagerr.txt', errors_diag)
