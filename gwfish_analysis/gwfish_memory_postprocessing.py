import os
import tqdm
import pickle

import lal

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer

import gwfish_utils as gu
import plotting_utils as pu
import GWFish.modules as gw

all_detectors = ['CE1','ET']
#all_detectors = ['ET','CE1']
detector_combinations = [['ET'],['ET','CE1']]

opts = gu.parse_commandline()

popdir, totaldir, namebase = gu.output_names(opts)

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

alternative_models = [('J_E', opts.j_e), ('J_E', 1 - opts.j_e), ('J_J', opts.j_j), ('J_J', 1 - opts.j_j)]

for dtc in all_detectors:
  parameters['SNR'+dtc] = np.zeros(len(parameters))

threshold_SNR = np.array([0., 9.])


no_result = []
list_logz_dicts = {''.join(dtcomb): [] for dtcomb in detector_combinations}

for kk in tqdm.tqdm(range(len(parameters))):

  outfile_name = namebase+'_'+str(kk)+'.pkl'
  parameter_values = parameters.iloc[kk]
  if os.path.exists(totaldir + outfile_name):
    with open(totaldir + outfile_name, 'rb') as fh:
      fm_object = pickle.load(fh)
  else:
    no_result.append(kk)
    continue

  # Saving the Fisher matrix prior to frequency range cut, for inspection
  fm_old = fm_object.fm
  errors_old, _ = gw.fishermatrix.invertSVD(fm_old)

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
    fm_object.update_fm()

    # PE errors
    errors, _ = gw.fishermatrix.invertSVD(fm_object.fm)

    # SNR
    # Assuming only one detector below, but for ET there are 3 components
    SNRs = gw.detection.SNR(network.detectors[0], fm_object.derivative.projection_at_parameters, duty_cycle=False)
    parameters['SNR'+dtc].iloc[kk] = np.sqrt(np.sum(SNRs ** 2))
    #import ipdb; ipdb.set_trace()
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
    em_comb[dtcomb_label], _ = gw.fishermatrix.invertSVD(fm_comb[dtcomb_label].to_numpy())
    em_comb[dtcomb_label] = pd.DataFrame(em_comb[dtcomb_label], columns=fisher_parameters, index=fisher_parameters)

    log_z_true = gu.log_z(em_comb[dtcomb_label], 0.0)
    # Models where we fix one parameter
    log_zs = {}
    newmu = {}
    newcov = {}
    newmu_2 = {}
    newcov_2 = {}
    for am in alternative_models:
      log_zs[str(am)], newcov[str(am)], newmu[str(am)], loglr = gu.log_z_alternative_model(am[0], am[1], em_comb[dtcomb_label], true_values, invcov=fm_comb[dtcomb_label])
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
      if extra is not None:
        try:
            newfisher, _ = gw.fishermatrix.invertSVD(newcov[str(am)])
            newfisher_matrix = pd.DataFrame(newfisher, columns=newcov[str(am)].columns, index=newcov[str(am)].index)
            log_zs[str(am)+str(extra)], newcov_2[str(am)], newmu_2[str(am)], loglr_2 = gu.log_z_alternative_model(extra[0], extra[1], newcov[str(am)], newmu[str(am)], invcov=newfisher_matrix, log_l_max=loglr)
        except:
            print('SVD error in extra memory parameter:', am, extra, kk, dtcomb)
    list_logz_dicts[dtcomb_label].append(log_zs)

  # Corner plot
  cc = ChainConsumer()
  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)
    chain_sim = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), em_comb[dtcomb_label], size=10000)
    cc.add_chain(chain_sim, parameters=fisher_parameters, name=dtcomb_label+': After 6.5*fisco cut')
  chain_sim_old = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors_old, size=10000)
  cc.add_chain(chain_sim_old, parameters=fisher_parameters, name='Before 6.5*fisco cut')
  cc.configure(usetex=False)
  fig = cc.plotter.plot()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_pe.png')
  plt.close()

  # Plot model misspecification
  cc = ChainConsumer()
  cc.add_chain(chain_sim, parameters=fisher_parameters, name='Correct model, J_E '+str(opts.j_e)+', J_J '+str(opts.j_j))
  for am in alternative_models:
    cc.add_chain(np.random.multivariate_normal(newmu[str(am)].to_numpy()[:,0], newcov[str(am)],size=10000), parameters=newmu[str(am)].index.to_list(), name=str(am)+', logBF^true_this='+str(log_z_true - log_zs[str(am)]))
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
    if extra is not None:
      cc.add_chain(np.random.multivariate_normal(newmu_2[str(am)].to_numpy()[:,0], newcov_2[str(am)],size=10000), parameters=newmu_2[str(am)].index.to_list(), name=str(am)+str(extra)+', logBF^true_this='+str(log_z_true - log_zs[str(am)+str(extra)]))

  ## Make a plot
  cc.configure(usetex=False)
  fig = cc.plotter.plot()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_'+dtcomb_label+'_pe_misspec.png')
  plt.close()

  # Save key numbers
  #import ipdb; ipdb.set_trace()

#import ipdb; ipdb.set_trace()

for dtcomb in detector_combinations:
  dtcomb_label = ''.join(dtcomb)
  dict_z = pd.DataFrame.from_dict(list_logz_dicts[dtcomb_label])
  
  je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
  je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
  je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
  
  je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
  
  ebms_vs_bms = dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key]
  ebms_vs_poincare = dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key]
  bms_vs_poincare = dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key]
  
  spinmonly_vs_poincare = dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key]
  
  # Plot sorted evidence for model combinations as a function of event number
  fig, ax = plt.subplots(figsize=(8, 5))
  indices = np.linspace(1,len(bms_vs_poincare),len(bms_vs_poincare))
  indices_major = np.arange(1,len(bms_vs_poincare),15)
  ax.plot(indices, bms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{BMS}}_{Poincare}$')
  ax.plot(indices, ebms_vs_poincare.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{Poincare}$')
  ax.plot(indices, ebms_vs_bms.sort_values(ascending=False).cumsum().values,label='$\ln \mathcal{B}^{\mathrm{eBMS}}_{BMS}$')
  ax.axhline(5., color='red', linestyle=':', label='Detection threshold')
  ax.legend()
  ax.set_xticks(indices_major)
  ax.set_xticks(indices, minor=True)
  ax.grid(which='both')
  ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$')
  ax.set_xlabel('Number of detections, sorted by evidence')
  plt.xlim([1,indices[-1]])
  plt.tight_layout()
  plt.savefig(totaldir+'_evidence_n_'+opts.label+'_'+dtcomb_label+'.png')
  plt.close()

import ipdb; ipdb.set_trace()

#jj=dict_z['(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)']
#je = dict_z['(\'J_E\', 1.0)'] - dict_z['(\'J_E\', 0.0)']
#jejj = dict_z['(\'J_E\', 1.0)(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)(\'J_E\', 0.0)']

# Show how log Bayes factor grows as a function of (significant) detections
