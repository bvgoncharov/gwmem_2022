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

#all_detectors = ['CE1','ET']
#all_detectors = ['ET','CE1']
all_detectors = ['ET','CE1','LLO','LHO','VIR','VOY']
#detector_combinations = [['ET'],['ET','CE1']]
detector_combinations = [['ET'],['ET','CE1'],['LLO','LHO','VIR'],['VOY','VIR']]
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
jejj_covs = {''.join(dtcomb): pd.DataFrame(columns=['JE','JJ','JEJJ'],index=parameters.iloc[range_simulations].index, dtype=np.float64) for dtcomb in detector_combinations}

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

  waveform_obj = fm_object.derivative.waveform_object
  fisco = 1.0 / (np.power(6.0, 1.5) * lal.PI *(waveform_obj._lal_mass_1 + waveform_obj._lal_mass_2) * lal.MTSUN_SI / lal.MSUN_SI)

  # Plot waveform
  #pu.plot_waveform(waveform_obj, totaldir+namebase+'_'+str(kk)+'_waveform.png', fisco)

  # Calculating PE errors for individual detectors
  error_matrix = {}
  fisher_matrix = {}
  true_values = pd.DataFrame(data=np.array([parameter_values[key] for key in fisher_parameters]), index=fisher_parameters)

  for dtc in all_detectors:

    # General
    network = gw.detection.Network([dtc], detection_SNR=threshold_SNR,
                                   parameters=parameters,
                                   fisher_parameters=fisher_parameters,
                                   config=opts.config)

    # Remove too-high frequencies with artefacts and re-calculate Fisher matrix
    #fm_object = gw.fishermatrix.FisherMatrix(fm_object.derivative.waveform_object, parameter_values, fisher_parameters, network.detectors[0])
    fm_object.detector = network.detectors[0]
    fm_object.derivative.detector = network.detectors[0]
    fm_object.detector.frequency_mask = np.squeeze(network.detectors[0].frequencyvector <= 6.5 * fisco)
    fm_object.derivative.projection_at_parameters = None # Removing default chached ET projection

    if network.detectors[0].name=='LISA':
        network.detectors[0].frequencyvector = waveform_obj.frequencyvector[:,np.newaxis]
        network.detectors[0].frequency_mask = np.squeeze(network.detectors[0].frequencyvector <= 6.5*waveform_obj.fisco)
        fm_object.detector = network.detectors[0]
        fm_object.derivative.detector = network.detectors[0]
        fm_object.detector.frequency_mask = np.squeeze(network.detectors[0].frequencyvector <= 6.5 * fisco)

    # This is for the case where we do not cache waveform parameters
    # In some LISA calculations I did, in some I did not (e.g. 1000 simulations)
    # It is much faster!
    if network.detectors[0].name!='LISA':
        fm_object.update_fm()

        # SNR
        # Assuming only one detector below, but for ET there are 3 components
        SNRs = gw.detection.SNR(network.detectors[0], fm_object.derivative.projection_at_parameters, duty_cycle=False)
        parameters['SNR'+dtc].iloc[kk] = np.sqrt(np.sum(SNRs ** 2))

    # PE errors
    errors, _ = gw.fishermatrix.invertSVD(fm_object.fm)

    # Model selection
    # Original model
    error_matrix[dtc] = pd.DataFrame(errors, columns=fisher_parameters, index=fisher_parameters)
    fisher_matrix[dtc] = pd.DataFrame(fm_object.fm, columns=fisher_parameters, index=fisher_parameters)

  # Calculating model misspecification, evidence, for detector combinations/networks
  em_comb = {}
  fm_comb = {}
  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)
    for dtc in dtcomb:
      if dtcomb_label in fm_comb.keys():
        fm_comb[dtcomb_label] += fisher_matrix[dtc]
      else:
        fm_comb[dtcomb_label] = fisher_matrix[dtc]
    em_comb[dtcomb_label], _ = gw.fishermatrix.invertSVD(fm_comb[dtcomb_label].to_numpy(), thresh=svd_threshold)
    em_comb[dtcomb_label] = pd.DataFrame(em_comb[dtcomb_label], columns=fisher_parameters, index=fisher_parameters)

    # Saving covariance
    jejj = em_comb[dtcomb_label][["J_E","J_J"]].loc[["J_E","J_J"]]
    jejj_covs[dtcomb_label]['JE'].loc[kk_idx] = jejj['J_E'].loc['J_E']
    jejj_covs[dtcomb_label]['JJ'].loc[kk_idx] = jejj['J_J'].loc['J_J']
    jejj_covs[dtcomb_label]['JEJJ'].loc[kk_idx] = jejj['J_E'].loc['J_J']

    #list_JEJJ_cov_dicts[dtcomb_label].append(em_comb[dtcomb_label][["J_E","J_J"]].loc[["J_E","J_J"]])

    log_z_true = gu.log_z(em_comb[dtcomb_label], 0.0)
    # Models where we fix one parameter
    log_zs = {}
    newmu = {}
    newcov = {}
    newmu_2 = {}
    newcov_2 = {}
    for am, labm, labe in zip(alternative_models,amod_labels,extra_labels):

      if '_' in labm: # This indicates a noise realization
        true_values = pd.DataFrame(data=np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), em_comb[dtcomb_label], size=1)[0,:], index=fisher_parameters)
      else:
        true_values = pd.DataFrame(data=np.array([parameter_values[key] for key in fisher_parameters]), index=fisher_parameters)

      log_zs[labm], newcov[labm], newmu[labm], loglr = gu.log_z_alternative_model(am[0], am[1], em_comb[dtcomb_label], true_values, invcov=fm_comb[dtcomb_label])
      extra = extra_model(am, opts)

      if extra is not None:
        try:
            newfisher, _ = gw.fishermatrix.invertSVD(newcov[labm], thresh=svd_threshold)
            newfisher_matrix = pd.DataFrame(newfisher, columns=newcov[labm].columns, index=newcov[labm].index)
            log_zs[labe], newcov_2[labe], newmu_2[labe], loglr_2 = gu.log_z_alternative_model(extra[0], extra[1], newcov[labm], newmu[labm], invcov=newfisher_matrix, log_l_max=loglr)
        except:
            log_zs[labe], newcov_2[labe], newmu_2[labe], loglr_2 = np.nan, np.nan, np.nan, np.nan
            print('SVD error in extra memory parameter:', am, extra, kk, dtcomb)

    #logz_detcombs[dtcomb_label].append(log_zs)
    logz_detcombs[dtcomb_label].loc[kk_idx] = log_zs

  # Corner plot
  #cc = ChainConsumer()
  #for dtcomb in detector_combinations:
  #  dtcomb_label = ''.join(dtcomb)
  #  chain_sim = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), em_comb[dtcomb_label], size=10000)
  #  cc.add_chain(chain_sim, parameters=fisher_parameters, name=dtcomb_label+': After 6.5*fisco cut')
  #chain_sim_old = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors_old, size=10000)
  #cc.add_chain(chain_sim_old, parameters=fisher_parameters, name='Before 6.5*fisco cut')
  #cc.configure(usetex=False)
  #fig = cc.plotter.plot()
  #plt.savefig(totaldir+namebase+'_'+str(kk)+'_pe.png')
  #plt.close()

  # Plot model misspecification
  #cc = ChainConsumer()
  #cc.add_chain(chain_sim, parameters=fisher_parameters, name='Correct model, J_E '+str(opts.j_e)+', J_J '+str(opts.j_j))
  #for am, labm, labe in zip(alternative_models, amod_labels, extra_labels):
  #  cc.add_chain(np.random.multivariate_normal(newmu[labm].to_numpy()[:,0], newcov[labm],size=10000), parameters=newmu[labm].index.to_list(), name=labm+', logBF^true_this='+str(log_z_true - log_zs[labm]))
  #  extra = extra_model(am, opts)

  #  if extra is not None:
  #    cc.add_chain(np.random.multivariate_normal(newmu_2[labm].to_numpy()[:,0], newcov_2[labe],size=10000), parameters=newmu_2[labe].index.to_list(), name=labe+', logBF^true_this='+str(log_z_true - log_zs[labe]))

  ## Make a plot
  #cc.configure(usetex=False)
  #fig = cc.plotter.plot()
  #plt.savefig(totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_pe_misspec.png')
  #plt.close()

  # Save key numbers
  #import ipdb; ipdb.set_trace()

#with open(totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_logzs.json','w') as fj:
#  json.dump(logz_detcombs, fj)
#print(totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_logzs.json')

for dtcomb in detector_combinations:
  dtcomb_label = ''.join(dtcomb)

  logzs_file_name = totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(opts.inj)+'_noise_'+str(opts.noise)+'_logzs.json'
  print('Saving',logzs_file_name)
  logz_detcombs[dtcomb_label].to_hdf(logzs_file_name, mode='w', key='root')

  jejj_cov_file_name = totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_'+str(opts.num)+'_'+str(opts.inj)+'_covjejj.json'
  print('Saving',jejj_cov_file_name)
  jejj_covs[dtcomb_label].to_hdf(jejj_cov_file_name, mode='w', key='root')

#for dtcomb in detector_combinations:
#  dtcomb_label = ''.join(dtcomb)
#  #dict_z = pd.DataFrame.from_dict(logz_detcombs[dtcomb_label])
#  dict_z = logz_detcombs[dtcomb_label]
#  
#  je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
#  je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
#  je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
#  
#  je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
#  
#  ebms_vs_bms = dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key]
#  ebms_vs_poincare = dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key]
#  bms_vs_poincare = dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key]
#  
#  spinmonly_vs_poincare = dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key]
#  
#  # Plot sorted evidence for model combinations as a function of event number
#  fig, ax = plt.subplots(figsize=(8, 5))
#  indices = np.linspace(1,len(bms_vs_poincare),len(bms_vs_poincare))
#  indices_major = np.arange(1,len(bms_vs_poincare),15)
#  ax.loglog(indices, bms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{BMS}}_{Poincare}$')
#  ax.loglog(indices, ebms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{Poincare}$')
#  ax.loglog(indices, ebms_vs_bms.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{BMS}$')
#  ax.axhline(5., color='red', linestyle=':', label='Detection threshold')
#  ax.legend()
#  ax.set_xticks(indices_major)
#  ax.set_xticks(indices, minor=True)
#  ax.grid(which='both')
#  ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$')
#  ax.set_xlabel('Number of detections, sorted by evidence')
#  plt.xlim([1,indices[-1]])
#  plt.tight_layout()
#  plt.savefig(totaldir+'_evidence_n_'+opts.label+'_'+dtcomb_label+'.png')
#  plt.close()

#import ipdb; ipdb.set_trace()

#jj=dict_z['(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)']
#je = dict_z['(\'J_E\', 1.0)'] - dict_z['(\'J_E\', 0.0)']
#jejj = dict_z['(\'J_E\', 1.0)(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)(\'J_E\', 0.0)']

# Show how log Bayes factor grows as a function of (significant) detections
