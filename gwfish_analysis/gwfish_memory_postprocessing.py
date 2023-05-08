import os
import tqdm
import pickle

import lal

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer

import gwfish_utils as gu
import GWFish.modules as gw

opts = gu.parse_commandline()

popdir, totaldir, namebase = gu.output_names(opts)

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

threshold_SNR = np.array([0., 9.])

# Matplotlib parameters
co = 'blue'
ls = '-'

no_result = []
list_logz_dicts = []

for kk in tqdm.tqdm(range(len(parameters))):
  outfile_name = namebase+'_'+str(kk)+'.pkl'
  parameter_values = parameters.iloc[kk]
  if os.path.exists(totaldir + outfile_name):
    with open(totaldir + outfile_name, 'rb') as fh:
      fm_object = pickle.load(fh)
  else:
    no_result.append(kk)
    continue

  # General
  network = gw.detection.Network([opts.det], detection_SNR=threshold_SNR,
                                 parameters=parameters,
                                 fisher_parameters=fisher_parameters,
                                 config=opts.config)
  waveform_obj = fm_object.derivative.waveform_object
  fisco = 1.0 / (np.power(6.0, 1.5) * lal.PI *(waveform_obj._lal_mass_1 + waveform_obj._lal_mass_2) * lal.MTSUN_SI / lal.MSUN_SI)

  # Re-calculate Fisher matrix?
  #for dd in np.arange(len(network.detectors)):
  #  fm_object.detector.frequency_mask = np.squeeze(network.detectors[dd].frequencyvector > fisco)
    # TO BE CONTINUED

    # This part is FisherMatrix.update_fm(), except that it 

  # PE errors
  errors, _ = gw.fishermatrix.invertSVD(fm_object.fm)

  # SNR
  # Assuming only one detector below, but for ET there are 3 components
  SNRs = gw.detection.SNR(network.detectors[0], fm_object.derivative.projection_at_parameters, duty_cycle=False)
  network.detectors[0].SNR[kk] = np.sqrt(np.sum(SNRs ** 2))

  # Plot waveform
  f_start, f_end = network.detectors[0].frequencyvector[0], network.detectors[0].frequencyvector[-1]
  hf = fm_object.derivative.waveform_at_parameters[0]
  fig, axs = plt.subplots(2,4, figsize=(20, 10), dpi=80)
  # Time-domain
  if hasattr(waveform_obj, 'lal_time_ht_plus'):
    axs[0,0].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+', color=co, linestyle=ls)
    axs[0,0].set_xlim([-0.01,0.01])
    axs[1,0].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx', color=co, linestyle=ls)
    axs[1,0].set_xlim([-0.15,0.15])
    axs[0,1].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+', color=co, linestyle=ls)
    #axs[0,1].set_xlim([-10,-1])
    axs[1,1].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx', color=co, linestyle=ls)
    axs[1,1].set_xlim([waveform_obj.lal_time_ht_cross[0],waveform_obj.lal_time_ht_cross[0]+100])
  ## Frequency-domain
  axs[0,2].loglog(network.detectors[0].frequencyvector, np.real(hf[:,0]), label='Re(h+)', color=co, linestyle=ls)
  #axs[0,2].set_xlim([f_start + 10,f_start + 20])
  axs[1,2].loglog(network.detectors[0].frequencyvector, np.real(hf[:,1]), label='Re(hx)', color=co, linestyle=ls)
  axs[1,2].set_xlim([f_start + 10,f_start + 20])
  axs[0,3].semilogx(network.detectors[0].frequencyvector, np.angle(hf[:,0] - 1j*hf[:,1]), label='Phase', alpha=0.3, color=co, linestyle=ls) # To be replace by np.angle()
  #axs[0,3].set_xlim([f_start + 10,f_start + 20])
  axs[1,3].loglog(network.detectors[0].frequencyvector, np.abs(hf[:,0] - 1j*hf[:,1]), label='Abs', color=co, linestyle=ls) # To be replace by np.angle()
  axs[1,3].axvline(x=4*fisco, label='4fisco', linestyle=':',color='grey')
  axs[1,3].axvline(x=6.5*fisco, label='6.5fisco', linestyle='--',color='grey')
  axs[0,0].set_xlabel('Time [s]')
  axs[1,0].set_xlabel('Time [s]')
  axs[0,1].set_xlabel('Time [s]')
  axs[1,1].set_xlabel('Time [s]')
  axs[0,0].set_ylabel('hp')
  axs[1,0].set_ylabel('hx')
  axs[0,1].set_ylabel('hp')
  axs[1,1].set_ylabel('hx')
  axs[0,2].set_xlabel('Frequency [Hz]')
  axs[1,2].set_xlabel('Frequency [Hz]')
  axs[0,3].set_xlabel('Frequency [Hz]')
  axs[1,3].set_xlabel('Frequency [Hz]')
  axs[0,2].set_ylabel('hp')
  axs[1,2].set_ylabel('hx')
  axs[0,3].set_ylabel('Complex strain phase')
  axs[1,3].set_ylabel('Complex strain amplitude') 
  plt.tight_layout()
  plt.legend()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_waveform.png')
  plt.close()

  # Plot memory
  #import ipdb; ipdb.set_trace()

  ## Corner plot
  cc = ChainConsumer()
  chain_sim = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors, size=10000)
  cc.add_chain(chain_sim, parameters=fisher_parameters)
  cc.configure(usetex=False)
  fig = cc.plotter.plot()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_pe.png')
  plt.close()

  # Model selection
  # Original model
  error_matrix = pd.DataFrame(errors, columns=fisher_parameters, index=fisher_parameters)
  fisher_matrix = pd.DataFrame(fm_object.fm, columns=fisher_parameters, index=fisher_parameters)
  true_values = pd.DataFrame(data=np.array([parameter_values[key] for key in fisher_parameters]), index=fisher_parameters)
  log_z_true = gu.log_z(error_matrix, 0.0)
  # Models where we fix one parameter
  alternative_models = [('J_E', opts.j_e), ('J_E', 1 - opts.j_e), ('J_J', opts.j_j), ('J_J', 1 - opts.j_j)]
  #alternative_models = [('mass_1', 35.6), ('mass_1', 35.7), ('mass_2', 30.6), ('mass_2', 30.7)]
  log_zs = {}
  cc = ChainConsumer()
  cc.add_chain(chain_sim, parameters=fisher_parameters, name='Correct model, J_E '+str(opts.j_e)+', J_J '+str(opts.j_j))
  for am in alternative_models:
    log_zs[str(am)], newcov, newmu, loglr = gu.log_z_alternative_model(am[0], am[1], error_matrix, true_values, invcov=fisher_matrix)
    cc.add_chain(np.random.multivariate_normal(newmu.to_numpy()[:,0], newcov,size=10000), parameters=newmu.index.to_list(), name=str(am)+', logBF^true_this='+str(log_z_true - log_zs[str(am)]))
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
          newfisher, _ = gw.fishermatrix.invertSVD(newcov)
          newfisher_matrix = pd.DataFrame(newfisher, columns=newcov.columns, index=newcov.index)
          log_zs[str(am)+str(extra)], newcov_2, newmu_2, loglr_2 = gu.log_z_alternative_model(extra[0], extra[1], newcov, newmu, invcov=newfisher_matrix, log_l_max=loglr)
          cc.add_chain(np.random.multivariate_normal(newmu_2.to_numpy()[:,0], newcov_2,size=10000), parameters=newmu_2.index.to_list(), name=str(am)+str(extra)+', logBF^true_this='+str(log_z_true - log_zs[str(am)+str(extra)]))
      except:
          print('SVD error in extra memory parameter:', am, extra, kk)
  list_logz_dicts.append(log_zs)
  # Obsolete part
  # Model where we fix one more parameter
  #extra = ('J_E', 1 - opts.j_e)
  #extra = ('mass_1', 35.6)
  #newfisher, _ = gw.fishermatrix.invertSVD(newcov)
  #newfisher_matrix = pd.DataFrame(newfisher, columns=newcov.columns, index=newcov.index)
  #log_z_2, newcov_2, newmu_2 = gu.log_z_alternative_model(extra[0], extra[1], newcov, newmu, invcov=newfisher_matrix)
  #log_z_3, newcov_3, newmu_3 = gu.log_z_alternative_model(extra[0], extra[1], newcov, newmu, invcov=newfisher_matrix)
  #cc.add_chain(np.random.multivariate_normal(newmu_2.to_numpy()[:,0], newcov_2,size=10000), parameters=newmu_2.index.to_list(), name=str(am)+str(extra)+', logBF^true_this='+str(log_z_true - log_z_2))

  ## Make a plot
  cc.configure(usetex=False)
  fig = cc.plotter.plot()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_pe_misspec.png')
  plt.close()

  # Save key numbers
  import ipdb; ipdb.set_trace()

import ipdb; ipdb.set_trace()
dict_z = pd.DataFrame.from_dict(list_logz_dicts)

je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]


ebms_vs_bms = dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key]
ebms_vs_poincare = dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key]
bms_vs_poincare = dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key]

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
plt.savefig(totaldir+'_evidence_n_'+opts.label+'.png')
plt.close()

import ipdb; ipdb.set_trace()

#jj=dict_z['(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)']
#je = dict_z['(\'J_E\', 1.0)'] - dict_z['(\'J_E\', 0.0)']
#jejj = dict_z['(\'J_E\', 1.0)(\'J_J\', 1.0)'] - dict_z['(\'J_J\', 0.0)(\'J_E\', 0.0)']

# Show how log Bayes factor grows as a function of (significant) detections
