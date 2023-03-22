import os
import tqdm
import pickle

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

for kk in tqdm.tqdm(range(len(parameters))):
  outfile_name = namebase+'_'+str(kk)+'.pkl'
  parameter_values = parameters.iloc[kk]
  if os.path.exists(totaldir + outfile_name):
    with open(totaldir + outfile_name, 'rb') as fh:
      fm_object = pickle.load(fh)
  else:
    no_result.append(kk)
    continue

  # PE errors
  errors, _ = gw.fishermatrix.invertSVD(fm_object.fm)

  # SNR
  network = gw.detection.Network([opts.det], detection_SNR=threshold_SNR, 
                                 parameters=parameters, 
                                 fisher_parameters=fisher_parameters, 
                                 config=opts.config)
  # Assuming only one detector below, but for ET there are 3 components
  SNRs = gw.detection.SNR(network.detectors[0], fm_object.derivative.projection_at_parameters, duty_cycle=False)
  network.detectors[0].SNR[kk] = np.sqrt(np.sum(SNRs ** 2))

  # Plot waveform
  waveform_obj = fm_object.derivative.waveform_object
  f_start, f_end = network.detectors[0].frequencyvector[0], network.detectors[0].frequencyvector[-1]
  hf = fm_object.derivative.waveform_at_parameters[0]
  fig, axs = plt.subplots(2,4, figsize=(20, 10), dpi=80)
  # Time-domain
  if hasattr(waveform_obj, 'lal_time_ht_plus'):
    axs[0,0].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+', color=co, linestyle=ls)
    axs[0,0].set_xlim([-0.01,0.01])
    axs[1,0].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx', color=co, linestyle=ls)
    axs[1,0].set_xlim([-0.01,0.01])
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

  # Corner plot
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
  #alternative_models = [('J_E', opts.j_e), ('J_E', 1 - opts.j_e), ('J_J', opts.j_j), ('J_J', 1 - opts.j_j)]
  alternative_models = [('mass_1', 35.6), ('mass_1', 35.7), ('mass_2', 30.6), ('mass_2', 30.7)]
  log_zs = {}
  cc = ChainConsumer()
  cc.add_chain(chain_sim, parameters=fisher_parameters, name='Correct model, J_E '+str(opts.j_e)+', J_J '+str(opts.j_j))
  for am in alternative_models:
      log_zs[str(am)], newcov, newmu = gu.log_z_alternative_model(am[0], am[1], error_matrix, true_values, invcov=fisher_matrix)
      cc.add_chain(np.random.multivariate_normal(newmu.to_numpy()[:,0], newcov,size=10000), parameters=newmu.index.to_list(), name=str(am)+', logBF^true_this='+str(log_z_true - log_zs[str(am)]))
  # Model where we fix one more parameter
  #extra = ('J_E', 1 - opts.j_e)
  extra = ('mass_1', 35.6)
  newfisher, _ = gw.fishermatrix.invertSVD(newcov)
  newfisher_matrix = pd.DataFrame(newfisher, columns=newcov.columns, index=newcov.index)
  log_z_2, newcov_2, newmu_2 = gu.log_z_alternative_model(extra[0], extra[1], newcov, newmu, invcov=newfisher_matrix)
  cc.add_chain(np.random.multivariate_normal(newmu_2.to_numpy()[:,0], newcov_2,size=10000), parameters=newmu_2.index.to_list(), name=str(am)+str(extra)+', logBF^true_this='+str(log_z_true - log_z_2))
  # Make a plot
  cc.configure(usetex=False)
  fig = cc.plotter.plot()
  plt.savefig(totaldir+namebase+'_'+str(kk)+'_pe_misspec.png')
  plt.close()

  # Save key numbers
  #import ipdb; ipdb.set_trace()
