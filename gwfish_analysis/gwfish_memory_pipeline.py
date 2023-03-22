import os
import json

import numpy as np
import pandas as pd

from numpy.random import default_rng

from itertools import combinations, chain

import GWFish.modules as gw

import gwfish_utils as gu

rng = default_rng()

def mismatch(ww1, ww2):
    return 1 - np.vdot(ww1, ww2) / np.vdot(ww1, ww1) ** 0.5 / np.vdot(ww2, ww2) ** 0.5

def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]

opts = gu.parse_commandline()

popdir, totaldir, namebase = gu.output_names(opts)

if not os.path.exists(totaldir):
    print('Output directory does not exist: ', totaldir)
    raise ValueError()

threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

np.random.seed(0)

waveform_class = eval(opts.waveform_class)

for kk in range(opts.inj*opts.num, (opts.inj+1)*opts.num):

    outfile_name = namebase+'_'+str(kk)+'.pkl'

    if kk==0:
        with open(totaldir + namebase + '_opts_.json', 'w') as fout:
            json.dump(opts.__dict__, fout, sort_keys=True, indent=4, separators=(',', ': '))
        # Save opts in a file

    parameter_values = parameters.iloc[kk]
    print(parameter_values)

    network = gw.detection.Network([opts.det], detection_SNR=threshold_SNR, parameters=parameters, fisher_parameters=fisher_parameters, config=opts.config)

    for dd in np.arange(len(network.detectors)):

        data_params = {
                          'frequencyvector': network.detectors[dd].frequencyvector,
                          #'frequency_mask': frequency_mask,
                          'memory_contributions': opts.mem_sim, #'J_E, J_J', # J_E, J_J
                          'time_domain_f_min': opts.td_fmin,
                          'f_ref': opts.f_ref
                      }

        waveform_obj = waveform_class(opts.waveform, parameter_values, data_params)

        network.detectors[dd].fisher_matrix[kk, :, :] = np.zeros((len(fisher_parameters),len(fisher_parameters)))

        fm_object = gw.fishermatrix.FisherMatrix(waveform_obj, parameter_values, fisher_parameters, network.detectors[dd])

        _ = fm_object.fm
        fm_object.pickle(totaldir + outfile_name)

print('Completed')
