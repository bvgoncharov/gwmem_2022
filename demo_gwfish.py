import numpy as np
import pandas as pd

from numpy.random import default_rng

import time
import json

from itertools import combinations, chain

from tqdm import tqdm

import argparse

import GWFish.modules as gw

rng = default_rng()

def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]


pop_file='../GWFish/injections/CBC_pop.hdf5'
pop_id='BBH'
detectors_ids=['ET']
networks='[[0]]'
config='../GWFish/GWFish/detectors.yaml'


threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
calculate_errors = True   # whether to calculate Fisher-matrix based PE errors
duty_cycle = False  # whether to consider the duty cycle of detectors

#fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2']
#fisher_parameters = ['luminosity_distance','ra','dec']

if networks == 'all':
    networks_ids = powerset(len(detectors_ids))
else:
    networks_ids = json.loads(networks)

parameters = pd.read_hdf(pop_file)

network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                               fisher_parameters=fisher_parameters, config=config)

# lisaGWresponse(network.detectors[0], frequencyvector)
# exit()

# horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
# exit()

#waveform_model = 'gwfish_TaylorF2'
# waveform_model = 'gwfish_IMRPhenomD'
#waveform_model = 'lalsim_TaylorF2'
#waveform_model = 'lalsim_IMRPhenomD'
#waveform_model = 'lalsim_IMRPhenomXPHM'
#waveform_model = 'lalsim_NRSur7dq4'
waveform_model = 'nrsur_NRHybSur3dq8'
#waveform_model = 'lalsim_IMRPhenomTHM' # time-domain HoM

np.random.seed(0)

print('Processing CBC population')
for k in tqdm(np.arange(1)):
    parameter_values = parameters.iloc[k]

    networkSNR_sq = 0
    for d in np.arange(len(network.detectors)):
        wave, t_of_f = gw.waveforms.hphc_amplitudes(waveform_model, parameter_values,
                                                    network.detectors[d].frequencyvector)
                                                    #plot=network.detectors[d].plotrange)
        signal = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)

        SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
        networkSNR_sq += np.sum(SNRs ** 2)
        network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))

        if calculate_errors:
            network.detectors[d].fisher_matrix[k, :, :] = \
                gw.fishermatrix.FisherMatrix(waveform_model, parameter_values, fisher_parameters, network.detectors[d])

    network.SNR[k] = np.sqrt(networkSNR_sq)

gw.detection.analyzeDetections(network, parameters, pop_id, networks_ids)

if calculate_errors:
    gw.fishermatrix.analyzeFisherErrors(network, parameters, fisher_parameters, pop_id, networks_ids)
