import os
import json
import optparse

import numpy as np
import pandas as pd

from numpy.random import default_rng

from itertools import combinations, chain

import GWFish.modules as gw

import gwfish_utils as gu

rng = default_rng()

def parse_commandline():
    """
    Parse the command-line options.
    """
    parser = optparse.OptionParser()

    parser.add_option("-i", "--inj", help="Injection pack",  default=0, type=int)
    parser.add_option("-n", "--num", help="Number of injections in a pack", 
                      default=1, type=int)
    parser.add_option("-f", "--injfile", help="Injection/population file", 
                      default=None, type=str)
    parser.add_option("-o", "--outdir", help="Output directory", default='./',
                      type=str)
    parser.add_option("-l", "--label", help="Label for output files", default='_',
                      type=str)
    parser.add_option("-w", "--waveform", help="Waveform name", 
                      default='NRHybSur3dq8', type=str)
    parser.add_option("-W", "--waveform_class", help="Waveform class",
                      default='gw.waveforms.LALFD_Waveform', type=str)
    parser.add_option("-d", "--det", help="Detectors", default='ET', 
                      type=str)
    #parser.add_option("-x", "--networks", help="Network IDs",
    #                  default='[[0]]', type=str)
    parser.add_option("-c", "--config", help="Detector configuration", 
                      default='./gwfish_detectors.yaml', type=str)
    parser.add_option("-p", "--fisher_pars", help="Fisher parameters",
                      default='mass_1,mass2', type=str)
    parser.add_option("-r", "--f_ref", help="Reference frequency",  default=20.,
                      type=float)
    parser.add_option("-m", "--td_fmin", help="Time-domain f_min",  default=3., 
                      type=float)
    parser.add_option("-M", "--mem_sim", help="Memory terms to include",
                      default='J_E, J_J', type=str)
    parser.add_option("-e", "--j_e", 
                      help="J_E multiplicator, disp. memory (between 0 and 1)", 
                      default=0., type=float)
    parser.add_option("-j", "--j_j", 
                      help="J_J multiplicator, spin memory (between 0 and 1)", 
                      default=0., type=float)

    opts, args = parser.parse_args()

    return opts

def mismatch(ww1, ww2):
    return 1 - np.vdot(ww1, ww2) / np.vdot(ww1, ww1) ** 0.5 / np.vdot(ww2, ww2) ** 0.5

def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]

def m_chirp_from_component_m(m1,m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

# For Bayes factors and systematic errors

def conditioned_gaussian(sigma, mu, mu_key, mu_value):
    """
    Calculate a slice of a multivariate Gaussian distribution by
    fixing its position along one dimension at any value (not 
    necessarily the mean value along this dimension).

    Reference: https://en.wikipedia.org/wiki/Schur_complement

    Inputut values are pandas DataFrame.

    Returns new sigma and mu.
    """
    schurA = sigma.loc[sigma.index != mu_key, sigma.columns != mu_key]
    schurB = sigma.loc[sigma.index != mu_key, sigma.columns == mu_key]
    schurC = sigma.loc[sigma.index == mu_key, sigma.columns != mu_key]
    schurD = sigma.loc[sigma.index == mu_key, sigma.columns == mu_key]

    newcov = schurA - schurB @ schurD**(-1) @ schurC

    newmu = mu.loc[mu.index != mu_key] + schurB @ schurD**(-1) @ \
            (mu_value - mu.loc[mu.index == mu_key])

    return newcov, newmu

def log_likelihood_ratio(inv_cov, offset):
    loglr = -0.5 * offset.transpose() @ inv_cov @ offset
    return loglr[0][0]

def log_z(covm, log_l_max):
    n_dim = len(covm.index)
    return n_dim / 2 * np.log(2*np.pi) + 0.5 * np.log(np.linalg.det(covm)) + log_l_max

def log_z_alternative_model(parameter, value, cov, mean, invcov=None):
    """
    Log evidence for an alternative model when parameter is fixed at value.
    For the likelihood described by mean (true values, maximum-likelihood) and cov.

    The result is (log_z - log_l_true), where log_l_true is the maximum
    likelihood value for the correct model. However, when finding log Bayes
    factor relative to the true model, log_l_true is cancelled out.

    Input:
    mean and cov: pandas.DataFrame
    parameter: str
    value: float
    """
    newcov, newmu = conditioned_gaussian(cov, mean, parameter, value)
    newmu_extended = pd.concat((newmu, pd.DataFrame([value],index=[parameter])))
    offset = mean - newmu_extended
    loglr = log_likelihood_ratio(invcov, offset)
    log_z_new = log_z(newcov, loglr)
    return log_z_new, newcov, newmu

opts = parse_commandline()

#pop_file = '../GWFish/injections/BBH_1e5.hdf5'
#pop_file = '../GWFish/injections/CBC_pop.hdf5'
#pop_file = opts.injfile

popdir = os.path.basename(opts.injfile).split('.')[0]
totaldir = opts.outdir + popdir + '/'
if not os.path.exists(totaldir):
    print('Output directory does not exist: ', totaldir)
    raise ValueError()

threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]

duty_cycle = False                  # whether to consider the duty cycle of detectors

fisher_parameters = opts.fisher_pars.split(',')

parameters = pd.read_hdf(opts.injfile)

#parameters.geocent_time = 0.0
#parameters.psi = 0.
#logging.warning('Hard-setting GW phase to pi/2, to avoid discrepancies between TD and SPH waveforms. Phase is not incorporated in TD waveforms, can be fixed in the future.')
#parameters.phase = np.pi/2 #+ np.pi/12 # TD waveforms are without phase, whereas phase is included for SPH waveforms. This hardcoding makes SPH and TD waveforms consistent
#parameters['a_1'] = np.repeat(0.1, len(parameters))
#parameters['a_2'] = np.repeat(0.1, len(parameters))
parameters['J_E'] = np.repeat(opts.j_e, len(parameters))
parameters['J_J'] = np.repeat(opts.j_j, len(parameters))

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

colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
linestyles = ['-','--',':', '-','--',':']
network = {}
np.random.seed(0)

# Obsolete
time_domain = [False, True]
#sph_modes = [False, False]

waveform_class = eval(opts.waveform_class)

print('Processing CBC population')

#waves = {}
#errors = {}
#fig, axs = plt.subplots(2,4, figsize=(20, 10), dpi=80)


#for wm, wc, co, ls in zip(waveform_models, waveform_class, colors, linestyles):

for kk in range(opts.inj*opts.num, (opts.inj+1)*opts.num):

    namebase = opts.label+'_'+opts.waveform+'_'+opts.waveform_class+'_'+str(opts.td_fmin)+'_'+str(opts.td_fmin)
    outfile_name = namebase+'_'+str(kk)+'.pkl'

    if kk==0:
        with open(totaldir + namebase + '_opts_.json', 'w') as fout:
            json.dump(opts.__dict__, fout, sort_keys=True, indent=4, separators=(',', ': '))
        # Save opts in a file

    parameter_values = parameters.iloc[kk]

    network = gw.detection.Network([opts.det], detection_SNR=threshold_SNR, parameters=parameters,
                               fisher_parameters=fisher_parameters, config=opts.config)

    kk = 0 # signal index
    parameter_values = parameters.iloc[kk]

    #networkSNR_sq = 0
    for dd in np.arange(len(network.detectors)):

        data_params = {
                          'frequencyvector': network.detectors[dd].frequencyvector,
                          #'frequency_mask': frequency_mask,
                          'memory_contributions': opts.mem_sim, #'J_E, J_J', # J_E, J_J
                          'time_domain_f_min': opts.td_fmin,
                          'f_ref': opts.f_ref
                      }

        waveform_obj = waveform_class(opts.waveform, parameter_values, data_params)

        #print('Processing', wm, '. Mismatch with IMRPhenomXPHM (real, imag): ', mismatch(waves[wm],waves[ref_model]).real, mismatch(waves[wm],waves[ref_model]).imag)

        network.detectors[dd].fisher_matrix[kk, :, :] = np.zeros((len(fisher_parameters),len(fisher_parameters)))

        fm_object = gw.fishermatrix.FisherMatrix(waveform_obj, parameter_values, fisher_parameters, network.detectors[dd])

        _ = fm_object.fm
        fm_object.pickle(totaldir + outfile_name)

        #network.detectors[dd].fisher_matrix[kk, :, :] += fm_object.fm

        #errors, _ = gw.fishermatrix.invertSVD(network.detectors[dd].fisher_matrix[kk, :, :])

        #SNRs = gw.detection.SNR(network.detectors[dd], fm_object.derivative.projection_at_parameters, duty_cycle=duty_cycle)
        #networkSNR_sq += np.sum(SNRs ** 2)
        #network.detectors[dd].SNR[kk] = np.sqrt(np.sum(SNRs ** 2))

    #f_start, f_end = network.detectors[dd].frequencyvector[0], network.detectors[dd].frequencyvector[-1]
    # Time-domain
    #if hasattr(waveform_obj, 'lal_time_ht_plus'):
    #    axs[0,0].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+', color=co, linestyle=ls)
    #    axs[0,0].set_xlim([-0.01,0.01])
    #    axs[1,0].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx', color=co, linestyle=ls)
    #    axs[1,0].set_xlim([-0.01,0.01])
    #    axs[0,1].plot(waveform_obj.lal_time_ht_plus, waveform_obj._lal_ht_plus.data.data, label='h+', color=co, linestyle=ls)
    #    #axs[0,1].set_xlim([-10,-1])
    #    axs[1,1].plot(waveform_obj.lal_time_ht_cross, waveform_obj._lal_ht_plus.data.data, label='hx', color=co, linestyle=ls)
    #    axs[1,1].set_xlim([waveform_obj.lal_time_ht_cross[0],waveform_obj.lal_time_ht_cross[0]+100])
    ## Frequency-domain
    #axs[0,2].loglog(network.detectors[d].frequencyvector, waves[:,0], label='h+', color=co, linestyle=ls)
    ##axs[0,2].set_xlim([f_start + 10,f_start + 20])
    #axs[1,2].loglog(network.detectors[d].frequencyvector, waves[:,1], label='hx', color=co, linestyle=ls)
    #axs[1,2].set_xlim([f_start + 10,f_start + 20])
    #axs[0,3].semilogx(network.detectors[d].frequencyvector, np.angle(waves[:,0] - 1j*waves[:,1]), label='Phase', alpha=0.3, color=co, linestyle=ls) # To be replace by np.angle()
    ##axs[0,3].set_xlim([f_start + 10,f_start + 20])
    #axs[1,3].loglog(network.detectors[d].frequencyvector, np.abs(waves[:,0] - 1j*waves[:,1]), label='Abs', color=co, linestyle=ls) # To be replace by np.angle()

    #network.SNR[kk] = np.sqrt(networkSNR_sq)
    #print('Network SNR: ', network.SNR[kk])

#axs[0,0].set_xlabel('Time [s]')
#axs[1,0].set_xlabel('Time [s]')
#axs[0,1].set_xlabel('Time [s]')
#axs[1,1].set_xlabel('Time [s]')
#axs[0,0].set_ylabel('hp')
#axs[1,0].set_ylabel('hx')
#axs[0,1].set_ylabel('hp')
#axs[1,1].set_ylabel('hx')
#
#axs[0,2].set_xlabel('Frequency [Hz]')
#axs[1,2].set_xlabel('Frequency [Hz]')
#axs[0,3].set_xlabel('Frequency [Hz]')
#axs[1,3].set_xlabel('Frequency [Hz]')
#axs[0,2].set_ylabel('hp')
#axs[1,2].set_ylabel('hx')
#axs[0,3].set_ylabel('Complex strain phase')
#axs[1,3].set_ylabel('Complex strain amplitude')
#
#plt.tight_layout()
#plt.legend()
#plt.savefig('../out_gwmem_2022/w.png')
#plt.close()

#if calculate_errors:
#    from chainconsumer import ChainConsumer
#    chains = {}
#    cc = ChainConsumer()
#    for wm, wc in zip(waveform_models, waveform_class):
#        new_key = wm + '_' + str(wc).split('.')[-1].split('\'')[0]
#        chains[new_key] = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[new_key],size=10000)
#        cc.add_chain(chains[new_key], parameters=fisher_parameters, name=new_key)
#    cc.configure(usetex=False)
#    fig = cc.plotter.plot()
#    plt.savefig('../out_gwmem_2022/w_err.png')
#    plt.close()
#
#    #chain_nrsur = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[other_waveform],size=10000)
#    #chain_xphm = np.random.multivariate_normal(np.array([parameter_values[key] for key in fisher_parameters]), errors[ref_model],size=10000)
#    #cc = ChainConsumer()
#    #cc.add_chain(chain_nrsur, parameters=fisher_parameters, name=other_waveform)
#    #cc.add_chain(chain_xphm, parameters=fisher_parameters, name=ref_model)
#    #cc.configure(usetex=False)
#    #fig = cc.plotter.plot()
#    #plt.savefig('../out_gwmem_2022/waveform_err_comparison.png')
#    #plt.close()

#from normal_corner import normal_corner
#figure_1 = normal_corner.normal_corner(errors['nrsur_NRHybSur3dq8'],np.array([parameter_values[key] for key in fisher_parameters]),fisher_parameters)
#plt.tight_layout()
#plt.savefig('../out_gwmem_2022/waveform_err_comparison.png')
#plt.close()

#if np.any(['Memory' in kk for kk in waveform_objects.keys()]):
#    fig, axs = plt.subplots(2,2, figsize=(20, 10), dpi=80)
#    for wok, wo in waveform_objects.items():
#        if 'Memory' in wok:
#            axs[0,0].plot(wo.lal_time_ht_plus, np.real(wo.J_J_modes[(3,0)]), label='SpinMem JJ (3,0) RE:'+wok)
#            axs[0,1].plot(wo.lal_time_ht_plus, np.real(wo.J_E_modes[(2,0)]), label='DispMem JE (2,0) RE:'+wok)
#            axs[1,0].plot(wo.lal_time_ht_plus, np.imag(wo.J_J_modes[(3,0)]), label='SpinMem JJ (3,0) IM:'+wok)
#            axs[1,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_E_modes[(2,0)]), label='DispMem JE (2,0) IM:'+wok) 
#    axs[0,0].legend()
#    axs[0,1].legend()
#    axs[1,0].legend()
#    axs[1,1].legend()
#    #fig.legend()
#    plt.tight_layout()
#    plt.savefig('../out_gwmem_2022/w_mem.png')
#    plt.close()
#
#    fig, axs = plt.subplots(2,2, figsize=(20, 20), dpi=80)
#    for wok, wo in waveform_objects.items():
#        if 'Memory' in wok:
#            for (ll,mm) in wo.J_J_modes:
#                axs[0,0].plot(wo.lal_time_ht_plus, np.real(wo.J_J_modes[(ll,mm)]), label='J_J RE '+str((ll,mm))+' '+wok)
#                axs[0,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_J_modes[(ll,mm)]), label='J_J IM '+str((ll,mm))+' '+wok)
#                axs[1,0].plot(wo.lal_time_ht_plus, np.real(wo.J_E_modes[(ll,mm)]), label='J_E RE '+str((ll,mm))+' '+wok)
#                axs[1,1].plot(wo.lal_time_ht_plus, np.imag(wo.J_E_modes[(ll,mm)]), label='J_E IM '+str((ll,mm))+' '+wok)
#    axs[0,0].legend()
#    axs[0,1].legend()
#    axs[1,0].legend()
#    axs[1,1].legend()
#    #fig.legend()
#    plt.tight_layout()
#    plt.savefig('../out_gwmem_2022/w_mem_modes.png')
#    plt.close()
#
#if False:
#    fig, axs = plt.subplots(2,1, figsize=(10, 10), dpi=80)
#    for wok, wo in waveform_objects.items():
#        if 'idx_low' in wo.__dict__.keys():
#            lal_vals = 1
#

#
#key = new_key # 'IMRPhenomXPHM_LALFD_Waveform' # or =new_key
#if False:
#    # Original model
#    error_matrix = pd.DataFrame(errors[key], columns=fisher_parameters, index=fisher_parameters)
#    fisher_matrix = pd.DataFrame(network[key].detectors[-1].fisher_matrix[0, :, :], columns=fisher_parameters, index=fisher_parameters) # -1 detector index, 0 signal index
#    true_values = pd.DataFrame(data=np.array([parameter_values[key] for key in fisher_parameters]), index=fisher_parameters)
#    log_z_true = log_z(error_matrix, 0.0)
#
#    # Models where we fix one parameter
#    alternative_models = [('J_E', 1.0), ('J_E', 0.0), ('J_J', 1.0), ('J_J', 0.0)]
#    #alternative_models = [('J_E', opts.j_e), ('J_E', 1 - opts.j_e), ('J_J', opts.j_j), ('J_J', 1 - opts.j_j)]
#    log_zs = {}
#    cc = ChainConsumer()
#    cc.add_chain(chains[key], parameters=fisher_parameters, name=key)
#    for am in alternative_models:
#        log_zs[str(am)], newcov, newmu = log_z_alternative_model(am[0], am[1], error_matrix, true_values, invcov=fisher_matrix)
#        cc.add_chain(np.random.multivariate_normal(newmu.to_numpy()[:,0], newcov,size=10000), parameters=newmu.index.to_list(), name=str(am)+', logBF^true_this='+str(log_z_true - log_zs[str(am)]))
#
#    # Model where we fix one more parameter
#    extra = ('J_E', 0.0)
#    #extra = ('J_E', 1 - opts.j_e)
#    newfisher, _ = gw.fishermatrix.invertSVD(newcov)
#    newfisher_matrix = pd.DataFrame(newfisher, columns=newcov.columns, index=newcov.index)
#    log_z_2, newcov_2, newmu_2 = log_z_alternative_model(extra[0], extra[1], newcov, newmu, invcov=newfisher_matrix)
#    cc.add_chain(np.random.multivariate_normal(newmu_2.to_numpy()[:,0], newcov_2,size=10000), parameters=newmu_2.index.to_list(), name=str(am)+str(extra)+', logBF^true_this='+str(log_z_true - log_zs[str(am)]))
#
#    # Make a plot
#    cc.configure(usetex=False)
#    fig = cc.plotter.plot()
#    plt.savefig('../out_gwmem_2022/w_err_misspec.png')
#    plt.close()
#
