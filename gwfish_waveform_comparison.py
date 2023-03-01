import logging

import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json

from itertools import combinations, chain

from tqdm import tqdm

import argparse

import GWFish.modules as gw

import gwfish_utils as gu

from matplotlib import pyplot as plt

rng = default_rng()

def mismatch(ww1, ww2):
    return 1 - np.vdot(ww1, ww2) / np.vdot(ww1, ww1) ** 0.5 / np.vdot(ww2, ww2) ** 0.5

def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]

def m_chirp_from_component_m(m1,m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)


#pop_file='../GWFish/injections/BBH_1e5.hdf5'
pop_file='../GWFish/injections/CBC_pop.hdf5'
pop_id='BBH'
detectors_ids=['ET']
networks='[[0]]'
config='./gwfish_detectors.yaml'


threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
calculate_errors = True             # whether to calculate Fisher-matrix based PE errors
duty_cycle = False                  # whether to consider the duty cycle of detectors

#fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2']
#fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2', 'a_1', 'a_2']
#fisher_parameters = ['ra', 'dec', 'psi', 'luminosity_distance'] # Parameters that worked initially for time-domain derivatives
#fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_chirp', 'mass_ratio']
#fisher_parameters = ['luminosity_distance','ra','dec']

if networks == 'all':
    networks_ids = powerset(len(detectors_ids))
else:
    networks_ids = json.loads(networks)

parameters = pd.read_hdf(pop_file)
parameters.mass_1 = 30.
parameters.mass_2 = 29.9999 #26.
#parameters.ra = 0.0
#parameters.dec = 0.0
parameters.theta_jn = np.pi/4 # np.pi/2 # 0.01
parameters.geocent_time = 0.0
#parameters.psi = 0.
logging.warning('Hard-setting GW phase to pi/2, to avoid discrepancies between TD and SPH waveforms. Phase is not incorporated in TD waveforms, can be fixed in the future.')
parameters.phase = np.pi/2 # TD waveforms are without phase, whereas phase is included for SPH waveforms. This hardcoding makes SPH and TD waveforms consistent
#parameters['a_1'] = np.repeat(0.5, len(parameters))
#parameters['a_2'] = np.repeat(0.5, len(parameters))

#parameters.luminosity_distance = 3000

# For m1,2 to M,q
#mass_chirp = m_chirp_from_component_m(parameters.mass_1, parameters.mass_2)
#mass_ratio = parameters.mass_1/parameters.mass_2
#parameters = parameters.rename(columns={'mass_1': 'mass_chirp', 'mass_2': 'mass_ratio'})
#parameters.mass_chirp = mass_chirp
#parameters.mass_ratio = mass_ratio


# lisaGWresponse(network.detectors[0], frequencyvector)
# exit()

# horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
# exit()
#ref_model = 'IMRPhenomXPHM'
#ref_model = 'IMRPhenomTPHM'
#ref_model = 'NRHybSur3dq8'
#waveform_model = 'gwfish_TaylorF2'
# waveform_model = 'gwfish_IMRPhenomD'
#waveform_model = 'lalsim_TaylorF2'
#waveform_model = 'lalsim_IMRPhenomD'
#waveform_model = 'lalsim_IMRPhenomXPHM'
#waveform_model = 'lalsim_NRSur7dq4'
#waveform_model = 'nrsur_NRHybSur3dq8'
#waveform_models = ['lalsim_IMRPhenomXPHM', 'nrsur_NRHybSur3dq8']
#other_waveform = 'lalsim_IMRPhenomTPHM'
#other_waveform = 'lalsim_NRHybSur3dq8'
#other_waveform = 'nrsur_NRHybSur3dq8'
#other_waveform = 'memes_NRHybSur3dq8'
#other_waveform = 'nrsur_NRSur7dq2'
#other_waveform = 'IMRPhenomTPHM'
#other_waveform = 'NRHybSur3dq8'
#waveform_models = ['lalsim_IMRPhenomXPHM', other_waveform]
#waveform_models = ['NRSur7dq4','NRSur7dq4','NRSur7dq4']
waveform_models = ['NRHybSur3dq8', 'NRHybSur3dq8', 'NRHybSur3dq8']
#waveform_models = ['IMRPhenomXPHM', 'IMRPhenomTPHM', 'NRHybSur3dq8']
#waveform_models = ['IMRPhenomTPHM', 'IMRPhenomTPHM', 'IMRPhenomTPHM']
colors = ['red', 'green', 'blue']
linestyles = ['-','--',':']
network = {}
np.random.seed(0)

# Obsolete
#time_domain = [False, True]
#sph_modes = [False, False]

#waveform_class = [gw.waveforms.LALFD_Waveform, gw.waveforms.LALTD_Waveform]
#waveform_class = [gw.waveforms.LALFD_Waveform, gu.LALTD_SPH_Waveform]
#waveform_class = [gw.waveforms.LALTD_Waveform, gu.LALTD_SPH_Waveform]
#waveform_class = [gu.LALTD_SPH_Waveform, gw.waveforms.LALTD_Waveform]
waveform_class = [gw.waveforms.LALFD_Waveform, gw.waveforms.LALTD_Waveform, gu.LALTD_SPH_Waveform]
#waveform_class = [gw.waveforms.LALFD_Waveform, gw.waveforms.LALFD_Waveform, gw.waveforms.LALFD_Waveform]
#waveform_class = [gw.waveforms.LALTD_Waveform, gu.LALTD_SPH_Waveform, gu.LALTD_SPH_Memory]
#waveform_class = [gu.LALTD_SPH_Memory, gu.LALTD_SPH_Waveform, gw.waveforms.LALTD_Waveform]
#waveform_class = [gu.LALTD_SPH_Waveform, gu.LALTD_SPH_Memory, gw.waveforms.LALTD_Waveform]
#waveform_class = [gu.LALTD_SPH_Waveform, gu.LALTD_SPH_Memory]
#waveform_class = [gw.waveforms.IMRPhenomD, gw.waveforms.LALFD_Waveform]

print('Processing CBC population')
waveform_objects = {}
waves = {}
errors = {}
fig, axs = plt.subplots(2,4, figsize=(20, 10), dpi=80)
for wm, wc, co, ls in zip(waveform_models, waveform_class, colors, linestyles):

    new_key = wm + '_' + str(wc).split('.')[-1].split('\'')[0]

    network[new_key] = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                               fisher_parameters=fisher_parameters, config=config)

    kk = 0
    parameter_values = parameters.iloc[kk]

    networkSNR_sq = 0
    for d in np.arange(len(network[new_key].detectors)):
        #original_fv = copy.copy(network[wm].detectors[d].frequencyvector)
        #waves[wm], t_of_f, frequencyvector

        data_params = {
                          'frequencyvector': network[new_key].detectors[d].frequencyvector,
                          #'frequency_mask': frequency_mask,
                          'memory_contributions': 'J_E, J_J', # J_E, J_J
                          #'f_ref': 50.
                      }
        waveform_obj = wc(wm, parameter_values, data_params)
        waveform_objects[new_key] = waveform_obj
        waves[new_key] = waveform_obj()
        t_of_f = waveform_obj.t_of_f

        #network[wm].detectors[d].frequencyvector = frequencyvector

        #print('Processing', wm, '. Mismatch with IMRPhenomXPHM (real, imag): ', mismatch(waves[wm],waves[ref_model]).real, mismatch(waves[wm],waves[ref_model]).imag)

        signal = gw.detection.projection(parameter_values, network[new_key].detectors[d], waves[new_key], t_of_f)
        #if wm == 'nrsur_NRHybSur3dq8':
        #    import ipdb; ipdb.set_trace()
        #SNRs = gw.detection.SNR(network[wm].detectors[d], signal, duty_cycle=duty_cycle)
        #networkSNR_sq += np.sum(SNRs ** 2)
        #network[wm].detectors[d].SNR[kk] = np.sqrt(np.sum(SNRs ** 2))

        if calculate_errors:
            network[new_key].detectors[d].fisher_matrix[kk, :, :] = \
                gw.fishermatrix.FisherMatrix(waveform_obj, parameter_values, fisher_parameters, network[new_key].detectors[d]).fm

            errors[new_key], _ = gw.fishermatrix.invertSVD(network[new_key].detectors[d].fisher_matrix[kk, :, :])

    f_start, f_end = network[new_key].detectors[d].frequencyvector[0], network[new_key].detectors[d].frequencyvector[-1]
    # Time-domain
    if hasattr(waveform_obj, 'lal_time_ht_plus'):
        axs[0,0].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+:'+new_key, color=co, linestyle=ls)
        axs[0,0].set_xlim([-0.01,0.01])
        axs[1,0].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx:'+new_key, color=co, linestyle=ls)
        axs[1,0].set_xlim([-0.01,0.01])
        axs[0,1].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+:'+new_key, color=co, linestyle=ls)
        #axs[0,1].set_xlim([-10,-1])
        axs[1,1].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx:'+new_key, color=co, linestyle=ls)
        axs[1,1].set_xlim([waveform_obj.lal_time_ht_cross[0],waveform_obj.lal_time_ht_cross[0]+100])
    # Frequency-domain
    axs[0,2].loglog(network[new_key].detectors[d].frequencyvector, waves[new_key][:,0], label='h+:'+new_key, color=co, linestyle=ls)
    axs[0,2].set_xlim([f_start + 10,f_start + 20])
    axs[1,2].loglog(network[new_key].detectors[d].frequencyvector, waves[new_key][:,1], label='hx:'+new_key, color=co, linestyle=ls)
    axs[1,2].set_xlim([f_start + 10,f_start + 20])
    axs[0,3].plot(network[new_key].detectors[d].frequencyvector, np.angle(waves[new_key][:,0] - 1j*waves[new_key][:,1]), label='Phase:'+new_key, alpha=0.3, color=co, linestyle=ls) # To be replace by np.angle()
    axs[0,3].set_xlim([f_start + 10,f_start + 20])
    axs[1,3].loglog(network[new_key].detectors[d].frequencyvector, np.abs(waves[new_key][:,0] - 1j*waves[new_key][:,1]), label='Abs:'+new_key, color=co, linestyle=ls) # To be replace by np.angle()

    network[new_key].SNR[kk] = np.sqrt(networkSNR_sq)
    print('Network SNR: ', network[new_key].SNR[kk])

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
plt.savefig('../out_gwmem_2022/w.png')
plt.close()

if calculate_errors:
    from chainconsumer import ChainConsumer
    chains = {}
    cc = ChainConsumer()
    for wm, wc in zip(waveform_models, waveform_class):
        new_key = wm + '_' + str(wc).split('.')[-1].split('\'')[0]
        chains[new_key] = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[new_key],size=10000)
        cc.add_chain(chains[new_key], parameters=fisher_parameters, name=new_key)
    cc.configure(usetex=False)
    fig = cc.plotter.plot()
    plt.savefig('../out_gwmem_2022/w_err.png')
    plt.close()

    #chain_nrsur = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[other_waveform],size=10000)
    #chain_xphm = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[ref_model],size=10000)
    #cc = ChainConsumer()
    #cc.add_chain(chain_nrsur, parameters=fisher_parameters, name=other_waveform)
    #cc.add_chain(chain_xphm, parameters=fisher_parameters, name=ref_model)
    #cc.configure(usetex=False)
    #fig = cc.plotter.plot()
    #plt.savefig('../out_gwmem_2022/waveform_err_comparison.png')
    #plt.close()

#from normal_corner import normal_corner
#figure_1 = normal_corner.normal_corner(errors['nrsur_NRHybSur3dq8'],np.array([parameter_values[key] for key in fisher_parameters]),fisher_parameters)
#plt.tight_layout()
#plt.savefig('../out_gwmem_2022/waveform_err_comparison.png')
#plt.close()

if np.any(['Memory' in kk for kk in waveform_objects.keys()]):
    fig, axs = plt.subplots(2,2, figsize=(20, 10), dpi=80)
    for wok, wo in waveform_objects.items():
        if 'Memory' in wok:
            axs[0,0].plot(wo.lal_time_ht_plus, np.real(wo.J_J_modes[(3,0)]), label='SpinMem JJ (3,0) RE:'+wok)
            axs[0,1].plot(wo.lal_time_ht_plus, np.real(wo.J_E_modes[(2,0)]), label='DispMem JE (2,0) RE:'+wok)
            axs[1,0].plot(wo.lal_time_ht_plus, np.imag(wo.J_J_modes[(3,0)]), label='SpinMem JJ (3,0) IM:'+wok)
            axs[1,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_E_modes[(2,0)]), label='DispMem JE (2,0) IM:'+wok) 
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    #fig.legend()
    plt.tight_layout()
    plt.savefig('../out_gwmem_2022/w_mem.png')
    plt.close()

    fig, axs = plt.subplots(2,2, figsize=(20, 20), dpi=80)
    for wok, wo in waveform_objects.items():
        if 'Memory' in wok:
            for (ll,mm) in wo.J_J_modes:
                axs[0,0].plot(wo.lal_time_ht_plus, np.real(wo.J_J_modes[(ll,mm)]), label='J_J RE '+str((ll,mm))+' '+wok)
                axs[0,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_J_modes[(ll,mm)]), label='J_J IM '+str((ll,mm))+' '+wok)
                axs[1,0].plot(wo.lal_time_ht_plus, np.real(wo.J_E_modes[(ll,mm)]), label='J_E RE '+str((ll,mm))+' '+wok)
                axs[1,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_E_modes[(ll,mm)]), label='J_E IM '+str((ll,mm))+' '+wok)
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    #fig.legend()
    plt.tight_layout()
    plt.savefig('../out_gwmem_2022/w_mem_modes.png')
    plt.close()

import ipdb; ipdb.set_trace()

#gw.detection.analyzeDetections(network, parameters, pop_id, networks_ids)
#
#if calculate_errors:
#    gw.fishermatrix.analyzeFisherErrors(network, parameters, fisher_parameters, pop_id, networks_ids)
