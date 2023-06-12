import os
import copy
import logging
import optparse

import numpy as np
import pandas as pd

import lal
import lalsimulation as lalsim

import GWFish.modules.waveforms as wf
import GWFish.modules.fft as fft

import gwtools
from gwtools import sxs_memory
import sxs

import time

#try:
#    import gwsurrogate
#    from scipy.signal.windows import tukey
#    from scipy import interpolate
#    #env_vars = os.environ
#    #if 'LAL_DATA_PATH' in env_vars:
#    #    sur = gwsurrogate.LoadSurrogate(os.environ['LAL_DATA_PATH']+'NRHybSur3dq8.h5')
#    #else:
#    #    raise ValueError('Please set LAL_DATA_PATH and put surrogate waveform files there.')
#    #import copy # only temporary
##except ModuleNotFoundError as err_gwsur:
##    print('Module gwsurrogate not found. Surrogate waveforms are not available.')

# FOR DEBUGGING
from matplotlib import pyplot as plt

# For pipeline

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
    parser.add_option("-l", "--label", help="Label for output files", \
                      default='_', type=str)
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
    parser.add_option("-s", "--save_waveforms", help="Save all waveforms \
                      evaluated during the Fisher matrix calculation",  \
                      default=1, type=int)
    parser.add_option("-M", "--mem_sim", help="Memory terms to include",
                      default='J_E, J_J', type=str)
    parser.add_option("-N", "--mem_neg_modes", help="Mem., pass neg. modes",
                      default=1, type=int)
    parser.add_option("-e", "--j_e",
                      help="J_E multiplicator, disp. memory (between 0 and 1)",
                      default=0., type=float)
    parser.add_option("-j", "--j_j",
                      help="J_J multiplicator, spin memory (between 0 and 1)",
                      default=0., type=float)
    # Post-processing only
    parser.add_option("-g", "--noise", help="0 noise off, otherwise N realizations", \
                      default=0, type=int)
    parser.add_option("-G", "--randomize_mem_pe", help="On/off noise for saved memory covariance matrices", \
                      default=0, type=int)
    parser.add_option("-t", "--svd_threshold", help="SVD inversion",  default=1e-10,
                      type=float)

    opts, args = parser.parse_args()

    return opts

def output_names(opts):
    popdir = os.path.basename(opts.injfile).split('.')[0]
    totaldir = opts.outdir + popdir + '/'
    namebase = opts.label+'_'+opts.waveform+'_'+opts.waveform_class+\
               '_'+str(opts.td_fmin)+'_'+str(opts.f_ref)
    return popdir, totaldir, namebase

# For model selection and analysis

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

def log_z_alternative_model(parameter, value, cov, mean, invcov=None, log_l_max=0.):
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
    log_l_max: float, assumed zero except when about to fix a parameter the 
        second time, then it is the value of the original likelihood at 
        the parameter fixed the first time.
    """
    newcov, newmu = conditioned_gaussian(cov, mean, parameter, value)
    newmu_extended = pd.concat((newmu, pd.DataFrame([value],index=[parameter])))
    offset = mean - newmu_extended
    loglr = log_likelihood_ratio(invcov, offset)
    log_z_new = log_z(newcov, log_l_max + loglr)
    return log_z_new, newcov, newmu, loglr

# For waveforms

def m_chirp_from_component_m(m1,m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def sxs_waveform_to_h_lm_dict(sxs_waveform, all_possible_modes):
    return {(ll, mm): np.array( sxs_waveform[:,sxs_waveform.index(ll,mm)] ) \
            for (ll, mm) in all_possible_modes}

def h_lm_dict_to_sxs_waveform(h_lm, times, mem_additional_ell=0):
    """ Create SXS WaveformModes objects from a dictionary of strain (l,m) modes.
    Based on gwtools.add_memory().

    Input
    =====
    h_lm:    dictionary of oscillatory modes. For example,
    
                 hdict_tmp[(2,2)] = np.array( [...] )
  
    mem_additional_ell: compute memory modes beyond ell_max found from h_lm
  
       Example: if the largest ell in h_lm is ell=3, and mem_additional_ell=2
                then compute memory modes up to ell=5.
  
       Note: compute time quickly goes up with ell!
    
    Output
    ======
    memory_sxs: dictionary of memory modes
    
    """
 
    # find the maximum/min value of ell stored in hdict_tmp

    #max_ell=-1
    #min_ell=100
    #last_mode = None
    #for mode in h_lm.keys():
    #  ell = mode[0]
    #  if ell > max_ell:
    #    max_ell = ell
    #  if ell < min_ell:
    #    min_ell = ell
    #  last_mode = mode
  
    #assert(last_mode is not None)

    # Above replaced by:
    max_ell = max(h_lm)[0]
    min_ell = min(h_lm)[0]

    # compute more or less memory modes than oscillatory modes
    max_ell = max_ell + mem_additional_ell
  
    modes = [(ell, m) for ell in range(2, max_ell+1) for m in range(-ell,ell+1)]

    sxs_modes = []
    for mode in modes:
      if mode in h_lm.keys():
        sxs_modes.append( h_lm[mode] )
      else:
        sxs_modes.append( np.zeros_like(h_lm[last_mode]) )
    sxs_modes = np.array( sxs_modes )
  
    h_test = sxs.waveforms.WaveformModes(sxs_modes.transpose(), time=times, modes_axis=1, time_axis=0, ell_min=min_ell, ell_max=max_ell)
    h_test._metadata['spin_weight'] = -2
  
    return h_test

# FFT
#from scipy.signal.windows import tukey
#
#def apply_window(waveform, times, alpha=0.2):
#    alpha = get_alpha(kwargs, times)
#    window = tukey(M=len(times), alpha=alpha)
#    for mode in waveform.keys():
#        waveform[mode] *= window
#    return waveform
#
#def fft(yy, dx, x_start, x_end, roll_off = 0.2):
#    """
#    Perform FFT to convert the data from time domain to frequency domain. 
#    Roll-off is specified for the Tukey window in [s].
#    """
#    alpha = 2 * roll_off / (x_end - x_start)
#    window = tukey(len(yy), alpha=alpha)
#    yy_tilde = np.fft.rfft(yy * window)
#    yy_tilde /= 1/dx
#    ff = np.linspace(0, (1/dx) / 2, len(yy_tilde))
#    # Future: here, one can check if frequency resolution and minimum frequency requested are
#    # lower than waveform time span. Resolution freq: warning. Minimum freq: ValueError.
#    return yy_tilde, ff, window
#
#def ifft(yy_tilde, df):
#    return np.fft.ifft(yy_tilde) * df

def fft_np(timeseries, dt, t_0, t_end):
    frequencyseries = np.fft.rfft(timeseries)
    frequencyseries /= 1/dt
    frequencies = np.linspace(0, (1/dt) / 2, len(frequencyseries))
    return frequencyseries, frequencies

def fft_wrapper(ht, timevector, delta_t, f_min, f_max, geocent_time):

    # FFT
    hf, ff = fft_np(ht, delta_t, timevector[0], timevector[-1])
    mask_f = (ff >= f_min) * (ff <= f_max)
    frequencyvector = ff[mask_f]
    delta_f = frequencyvector[1] - frequencyvector[0]
    hf = hf[mask_f]

    # Phase correction by epoch and delta t
    dt = 1/delta_f + timevector[0]
    hf *= np.exp(-1j * 2 * np.pi * dt * frequencyvector)

    # Phase correction by geocentric time
    phi_in = np.exp(1.j*(2*frequencyvector*np.pi*geocent_time))
    hf = phi_in * np.conjugate(hf)
 
    return hf, frequencyvector

# Gaussian noise generation

def gaussian_noise_fd(psd, delta_f, n_realiz=1):
    """
    Based on pycbc.noise.gaussian.frequency_noise_from_psd
    """
    sigma = 0.5 * (psd / delta_f) ** (0.5)
    noise_re = np.random.normal(0, sigma, (len(sigma),n_realiz))
    noise_im = np.random.normal(0, sigma, (len(sigma),n_realiz))
    return noise_re + 1j * noise_im

# For conversion to/from dimensionless units
def amp_rescale_coeff(dist, m_tot):
    """
    For conversion of gravitational-wave strain amplitude to/from 
    dimensionless units (DU). Multiply strain in DU by this value
    to obtain strain in regular MKS units.
    """
    return ((m_tot * gwtools.MSUN_SI ) / (1.e6*dist*gwtools.PC_SI )) * ( gwtools.G / np.power(gwtools.c,2.0) )

def time_rescale_coeff(m_tot):
    """
    For conversion of time corresponding to gravitational-wave strain
    in DU to/from MKS units.
    """
    return gwtools.Msuninsec * m_tot

# Custom waveform classes

class LALTD_SPH_Waveform(wf.LALTD_Waveform):
    """
    Calls SPH modes from LAL in time domain, sums modes, computes frequency-domain waveform.
    """
    def __init__(self, name, gw_params, data_params):
        self.l_max = None
        self._time_domain_f_min = None
        super(LALTD_SPH_Waveform, self).__init__(name, gw_params, data_params)
        self._lal_hf_plus = None
        self._lal_hf_cross = None
        self.idx_low = None
        self.idx_high = None

    @property
    def l_max(self):
        if self._l_max is None:
            if 'l_max' in self.data_params:
                self.l_max = self.data_params['l_max']
            else:
                self.l_max = 4
                logging.warning('Setting l_max to {}'.format(self._l_max))
        return self._l_max

    @l_max.setter
    def l_max(self, new_l_max):
        self._l_max = new_l_max

    @property
    def time_domain_f_min(self):
        """
        From fundamental Fourier transform relations, for frequency series with 
        f_max and delta_f, we need to have data duration: 
        T = 2 * f_max * delta_t / delta_f. 
        So, we need to have a time-domain waveform longer than that. This is 
        why f_min that we choose may be different from this required f_min for 
        time-domain data FFT.
        """
        if self._time_domain_f_min is None:
            if 'time_domain_f_min' in self.data_params:
                self._time_domain_f_min = self.data_params['time_domain_f_min']
            else:
                self._time_domain_f_min = 3
                logging.warning('Setting l_max to {}'.format(\
                                self._time_domain_f_min))
        return self._time_domain_f_min

    @property
    def _lal_mass_1(self):
        return self.gw_params['mass_1'] * lal.MSUN_SI * (1 + self.gw_params['redshift'])

    @property
    def _lal_mass_2(self):
        return self.gw_params['mass_2'] * lal.MSUN_SI * (1 + self.gw_params['redshift'])

    # [NOTE] Commented-out text is for when frequency vector has changed
    # In principle, this should not happen. For LAL FFT, only specific vectors
    # are required. So, one should show a warning.

    #@property
    #def frequencyvector(self):
    #    if self._lal_hf_plus is not None and \
    #            len(self._frequencyvector) != self.idx_high+1-self.idx_low: #self._lal_hf_plus.data.length != len(self._frequencyvector):
    #        self._update_frequency_range_indices()
    #        self._frequencyvector = np.arange(0, self.delta_f*self._lal_hf_plus.data.length, 
    #                                         self.delta_f)
    #        self._frequencyvector = self._frequencyvector[self.idx_low:self.idx_high+1]
    #    return self._frequencyvector

    #@frequencyvector.setter
    #def frequencyvector(self, new_frequencyvector):
    #    self._frequencyvector = np.squeeze(new_frequencyvector)

    #@property
    #def delta_f(self):
    #    if self._lal_hf_plus is not None:
    #        return self._lal_hf_plus.deltaF
    #    else:
    #        return self._frequencyvector[1] - self._frequencyvector[0]

    #def _lal_fd_strain_adjust_frequency_range(self):
    #    """ This needs to be changed compared to gwfish.LALFD_Waveform, delta_f is different """
    #    idx_low = int(self.f_min / self.delta_f)
    #    idx_high = int(self.f_max / self.delta_f)
    #    self.hf_cross_out = self._lal_hf_cross.data.data[idx_low:idx_high+1]
    #    self.hf_plus_out = self._lal_hf_plus.data.data[idx_low:idx_high+1]

    def _setup_lal_caller_args(self):
        """
        Dev. note: lalsim.SimInspiralChooseTDModes provides unconditioned time-domain mode data.
        The data is conditioned in lalsim.SimInspiralModesTD, but spin parameters are 
        not passed there. So, if spin are not needed, lalsim.SimInspiralModesTD
        might be a better option.
        """
        if lalsim.SimInspiralImplementedTDApproximants(self._approx_lal):
            self._waveform_postprocessing = self._ht_postproccessing_SimInspiralCTDM
            #self._waveform_postprocessing = self._ht_postproccessing_SimInspiralTD
            self._lalsim_caller = lalsim.SimInspiralChooseTDModes # general
            #self._lalsim_caller = lalsim.SimInspiralModesTD # no spin!



            # NEW: for data conditioning
            extra_time_fraction = 0.1
            extra_cycles = 3.0
            textra = extra_cycles / self.f_min
            tchirp = lalsim.SimInspiralChirpTimeBound(self.f_min, self._lal_mass_1, self._lal_mass_2, self.gw_params['spin_1z'], self.gw_params['spin_2z'])
            ss = lalsim.SimInspiralFinalBlackHoleSpinBound(self.gw_params['spin_1z'], self.gw_params['spin_2z'])
            tmerge = lalsim.SimInspiralMergeTimeBound(self._lal_mass_1, self._lal_mass_2) + lalsim.SimInspiralRingdownTimeBound(self._lal_mass_1 + self._lal_mass_2, ss)
            self.dc_fstart = lalsim.SimInspiralChirpStartFrequencyBound((1.0 + extra_time_fraction) * tchirp + tmerge + textra, self._lal_mass_1, self._lal_mass_2)



            self._lalsim_args = [
                0, # phiRef, unused parameter
                self.delta_t,
                self._lal_mass_1,
                self._lal_mass_2,
                self.gw_params['spin_1x'], self.gw_params['spin_1y'], self.gw_params['spin_1z'],
                self.gw_params['spin_2x'], self.gw_params['spin_2y'], self.gw_params['spin_2z'],
                self.dc_fstart, # self.time_domain_f_min, # NEW: replaced fmin for data cond.
                self.f_ref,
                self.gw_params['luminosity_distance'] * lal.PC_SI * 1e6,  # in [m]
                self._params_lal,
                self.l_max,
                self._approx_lal
            ]
        else:
            raise ValueError('Waveform approximant is not implemented in time-domain in LALSimulation.')

    def match_sph_duration_to_t_obs(self, ts):
        """
        ts: time series data from LALSimulation
        """
        required_length = int(self.t_obs / self.delta_t)
        if self.t_obs % self.delta_t != 0:
            raise ValueError('self.t_obs=1/self.delta_f should be a multiple of self.delta_t.')
        #logging.warning('Rounding up time vectors. An error/warning should be given here if t_obs is not fold of delta_t')
        ts = lal.ResizeCOMPLEX16TimeSeries(ts, ts.data.length-required_length, required_length)
        return ts

    def calculate_time_domain_sph_modes(self):
        # Note, waveform below is already conditioned (tapered)
        self._lal_hlms_iter = self._lalsim_caller(*self._lalsim_args)
        self._lal_hlms = {}
        while self._lal_hlms_iter:
            self._lal_hlms[self._lal_hlms_iter.l, self._lal_hlms_iter.m] = self._lal_hlms_iter.mode
            self._lal_hlms[self._lal_hlms_iter.l, self._lal_hlms_iter.m] = self.match_sph_duration_to_t_obs(self._lal_hlms[self._lal_hlms_iter.l, self._lal_hlms_iter.m])
            self._lal_hlms_iter = self._lal_hlms_iter.next

        # Work in progress
        #from gwtools import sxs_memory
        #from matplotlib import pyplot as plt
        #import tqdm
        #t_of_f = np.arange(0, self._lal_hlms[(5,5)].deltaT*self._lal_hlms[(5,5)].data.length, self._lal_hlms[(5,5)].deltaT)
        #hlms = {kk: vv.data.data for kk, vv in self._lal_hlms.items()}
        #h_mem_sxs, times_sxs = sxs_memory(hlms, t_of_f)
        #plt.close()
        #for kk in tqdm.tqdm(hlms.keys()):
        #    plt.plot(t_of_f, np.real(hlms[kk]),label='Before memory '+str(kk))
        #    plt.plot(times_sxs, np.real(h_mem_sxs[kk]), label='After memory '+str(kk))
        #    plt.xlabel('t')
        #    plt.ylabel('h')
        #    plt.legend()
        #    plt.savefig('/Users/boris.goncharov/projects/out_gwmem_2022/mem_modes_test/'+self.name+'_'+str(kk)+'.png')
        #    plt.close()

    def _td_strain_from_sph_modes(self):
        _m = (2,2)
        self._lal_ht_plus = lal.CreateREAL8TimeSeries('TD h-plus', 
                                                      self._lal_hlms[_m].epoch, 
                                                      self._lal_hlms[_m].f0, 
                                                      self._lal_hlms[_m].deltaT, 
                                                      self._lal_hlms[_m].sampleUnits, 
                                                      self._lal_hlms[_m].data.length)
        self._lal_ht_plus.data.data = np.zeros(self._lal_ht_plus.data.length)
        self._lal_ht_cross = lal.CreateREAL8TimeSeries('TD h-cross',
                                                      self._lal_hlms[_m].epoch,
                                                      self._lal_hlms[_m].f0,
                                                      self._lal_hlms[_m].deltaT,
                                                      self._lal_hlms[_m].sampleUnits,
                                                      self._lal_hlms[_m].data.length)
        self._lal_ht_cross.data.data = np.zeros(self._lal_ht_cross.data.length)

        fake_neg_modes = not np.any([mm < 0 for (ll, mm) in self._lal_hlms])

        for (ll, mm) in self._lal_hlms:
            # To test modes, not a good idea to use because of possible confusion
            #if ll > self.l_max:
            #    continue
            ylm = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], # inclination 
                                                    self.gw_params['phase'], -2, ll, mm)
            # LAL: Cross-polarization is the *negative* of the imaginary part
            self._lal_ht_plus.data.data += np.real(ylm * self._lal_hlms[(ll, mm)].data.data)
            self._lal_ht_cross.data.data -= np.imag(ylm * self._lal_hlms[(ll, mm)].data.data)
            # If m<0 modes are not in the dictionary of modes, we calculate
            # them from m>0 modes
            if fake_neg_modes and mm>0:
                print('BG: faking negative modes td_strain_from_sph')
                yl_m = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'],
                                                        self.gw_params['phase'], -2, ll, -mm)
                self._lal_ht_plus.data.data += np.real(yl_m * (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data))
                self._lal_ht_cross.data.data -= np.imag(yl_m * (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data))
        # Here, waveform is not conditioned

    def calculate_time_domain_strain(self):
        self.calculate_time_domain_sph_modes()
        self._td_strain_from_sph_modes()
        self._waveform_postprocessing()


    def _taper_lal(self):
        """
        Data conditioning: option 1, that might change fRef.
        This is based on SimInspiralTDfromTD.
        """

        extra_time_fraction = 0.1
        extra_cycles = 3.0
        textra = extra_cycles / self.f_min

        tchirp = lalsim.SimInspiralChirpTimeBound(self.f_min, 
            self._lal_mass_1,
            self._lal_mass_2,
            self.gw_params['spin_1z'], self.gw_params['spin_2z'])

        fisco = 1.0 / (np.power(6.0, 1.5) * lal.PI * \
                (self._lal_mass_1 + self._lal_mass_2) * \
                lal.MTSUN_SI / lal.MSUN_SI)

        # Window
        #self._window_p = lal.CreateREAL8TimeSeries(
        #        'window_htp',
        #        self._lal_ht_plus.epoch,
        #        self._lal_ht_plus.f0,
        #        self._lal_ht_plus.deltaT,
        #        self._lal_ht_plus.sampleUnits,
        #        self._lal_ht_plus.data.length
        #)
        ##self._window_p.data.data = np.ones(self._window_p.data.length)
        #self._window_p.data.data = self._lal_ht_plus.data.data
        #self._window_c = lal.CreateREAL8TimeSeries(
        #        'window_htp',
        #        self._lal_ht_cross.epoch,
        #        self._lal_ht_cross.f0,
        #        self._lal_ht_cross.deltaT,
        #        self._lal_ht_cross.sampleUnits,
        #        self._lal_ht_cross.data.length
        #)
        ##self._window_c.data.data = np.ones(self._window_c.data.length)
        #self._window_c.data.data = self._lal_ht_cross.data.data
        # Saving window function variable to correct for its effect later
        #self._window_p = copy.copy(self._lal_ht_plus)
        #self._window_p.data.data = np.ones(self._lal_ht_plus.data.length)
        #self._window_c = copy.copy(self._lal_ht_cross)
        #self._window_c.data.data = np.ones(self._lal_ht_cross.data.length)

        # Ignoring zeros at the beginning of LAL time series
        mask_zeros = self._lal_ht_plus.data.data != 0
        # Tapering beginning of the waveform, several cycles
        self._lal_ht_plus.data.data[mask_zeros] = taper_1(self._lal_ht_plus.data.data[mask_zeros], self._lal_ht_plus.deltaT, extra_time_fraction * tchirp + textra)
        self._lal_ht_cross.data.data[mask_zeros] = taper_1(self._lal_ht_cross.data.data[mask_zeros], self._lal_ht_cross.deltaT, extra_time_fraction * tchirp + textra)
        # Tapering one cycle at f_min at the beginning and one cycle at f_isco
        # at the end
        self._lal_ht_plus.data.data[mask_zeros] = taper_2(self._lal_ht_plus.data.data[mask_zeros], self._lal_ht_plus.deltaT, self.time_domain_f_min, fisco)
        self._lal_ht_cross.data.data[mask_zeros] = taper_2(self._lal_ht_cross.data.data[mask_zeros], self._lal_ht_cross.deltaT, self.time_domain_f_min, fisco)

        # Calling LAL data conditioning (does not work, for some reason)
        #lalsim.SimInspiralTDConditionStage1(self._lal_ht_plus, self._lal_ht_cross, extra_time_fraction * tchirp + textra, self.time_domain_f_min)
        #lalsim.SimInspiralTDConditionStage2(self._lal_ht_plus, self._lal_ht_cross, self.time_domain_f_min, fisco)

    def _ht_postproccessing_SimInspiralCTDM(self):
        self._taper_lal()
        self._ht_postproccessing_SimInspiralTD()


def taper_1(timeseries, delta_t, window_length):
    """
    Similar to SimInspiralTDConditionStage1, but without high-pass filtering.
    """
    ntaper = round(window_length / delta_t)
    idx = np.arange(0,len(timeseries),1)
    window = np.ones(len(timeseries))
    window[0:ntaper] = 0.5 - 0.5 * np.cos(idx[0:ntaper] * np.pi / ntaper)

    return timeseries * window

def taper_2(timeseries, delta_t, f_min, f_max):
    """
    Based on SimInspiralTDConditionStage2.
    """
    ntaper_end = round(1.0 / (f_max * delta_t))
    ntaper_start = round(1.0 / (f_min * delta_t))

    idx = np.arange(0,len(timeseries),1)
    window = np.ones(len(timeseries))

    window[-1:-1-ntaper_end:-1] = 0.5 - 0.5 * np.cos(idx[0:ntaper_end] * np.pi / ntaper_end)
    window[0:ntaper_start] = 0.5 - 0.5 * np.cos(idx[0:ntaper_start] * np.pi / ntaper_start)

    return timeseries * window

class LALTD_SPH_Memory(LALTD_SPH_Waveform):
    def __init__(self, name, gw_params, data_params):
        self.l_max = None
        self._time_domain_f_min = None
        super(LALTD_SPH_Waveform, self).__init__(name, gw_params, data_params)
        self._sxs_hlms_du = None
        self._J_J_modes = None
        self._J_E_modes = None
        #self.J_E = (0., 0.) # plus, cross
        #self.J_J = (0., 0.)
        self._fd_memory = 0.

    def update_gw_params(self, new_gw_params):
        self._update_gw_params_common(new_gw_params)
        # Specific to LALFD_Waveform and LALTD_Waveform
        self._init_lambda()
        self._init_lal_gw_parameters()
        self._setup_lal_caller_args()
        # Specific to LALTD_SPH_Memory (this class)
        self._J_J_modes = None
        self._J_E_modes = None
        self._sxs_hlms_du = None
        #self.J_E = (0., 0.)
        #self.J_J = (0., 0.)
        self._fd_memory = 0.

    def calculate_time_domain_strain(self):
        self.calculate_time_domain_sph_modes()
        self._add_missing_lms_in_lal_hlms()
        self._add_memory()
        self._td_strain_from_sph_modes()
        self._waveform_postprocessing()

    @property
    def m_tot(self):
        return self.gw_params['mass_1'] + self.gw_params['mass_2']

    @property
    def amp_rescale(self):
        """
        A multiplicative coefficient to convert GW time series data to/from 
        dimensionless units.
        """
        return amp_rescale_coeff(self.gw_params['luminosity_distance'], \
                                 self.m_tot*(1+self.gw_params['redshift']))

    @property
    def t_rescale(self):
        """
        A multiplicative coefficient to convert time array corresponding to 
        GW time series data to/from dimensionless units.
        """
        return time_rescale_coeff(self.m_tot*(1+self.gw_params['redshift']))

    @property
    def fd_memory(self):
        return None

    #def _fake_negative_sph_modes(self):
    #    fake_neg_modes = not np.any([mm < 0 for (ll, mm) in self._lal_hlms])
    #    for (ll, mm) in self._lal_hlms:

    #    (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data)

    def _add_memory(self):
        _len = self._lal_hlms[(2,2)].data.length
        fake_neg_modes = not np.any([mm < 0 for (ll, mm) in self.possible_modes])
        if self.calculate_J_J_modes:
            self.J_J = [np.zeros(_len), np.zeros(_len)]
            for lm in self.possible_modes:
                _strain_lm = self.J_J_modes[lm] * self.amp_rescale * self.gw_params['J_J']
                self._lal_hlms[lm].data.data += _strain_lm
                # Saving memory, also combining modes at the same time
                # (_td_strain_from_sph_modes)
                # It should be done simultaneously everywhere to avoid for loops
                ylm = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], self.gw_params['phase'], -2, lm[0], lm[1])
                self.J_J[0] += np.real(ylm * _strain_lm)
                self.J_J[1] -= np.imag(ylm * _strain_lm)
                if fake_neg_modes and mm>0:
                    print('BG: faking negative modes J_J')
                    yl_m = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], self.gw_params['phase'], -2, lm[0], -lm[1])
                    self.J_J[0] += np.real(yl_m * (-1)**(ll) * np.conjugate(_strain_lm))
                    self.J_J[1] -= np.imag(yl_m * (-1)**(ll) * np.conjugate(_strain_lm))
        if self.calculate_J_E_modes:
            self.J_E = [np.zeros(_len), np.zeros(_len)]
            for lm in self.possible_modes:
                _strain_lm = self.J_E_modes[lm] * self.amp_rescale * self.gw_params['J_E']
                self._lal_hlms[lm].data.data += _strain_lm
                # Saving memory, also combining modes at the same time
                # (_td_strain_from_sph_modes)
                # It should be done simultaneously everywhere to avoid for loops
                ylm = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], self.gw_params['phase'], -2, lm[0], lm[1])
                self.J_E[0] += np.real(ylm * _strain_lm)
                self.J_E[1] -= np.imag(ylm * _strain_lm)
                if fake_neg_modes and mm>0:
                    print('BG: faking negative modes J_E')
                    yl_m = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], self.gw_params['phase'], -2, lm[0], -lm[1])
                    self.J_E[0] += np.real(yl_m * (-1)**(ll) * np.conjugate(_strain_lm))
                    self.J_E[1] -= np.imag(yl_m * (-1)**(ll) * np.conjugate(_strain_lm))


        ## Old way, based on gwtools.sxs_memory:
        ##self.new_modes = 0
        #for lm in self.memory_modes.keys():
        #    if lm in self._lal_hlms:
        #        self._lal_hlms[lm].data.data += self.memory_modes[lm]
        #    else:
        #        logging.warning('Not summing negative m modes, check implications (something for negative frequencies, for precession).')
        #        #self._lal_hlms[lm] = self.memory_modes[lm]
        #        #self.new_modes += 1
        ##print('Number of new SPH modes after memory calculation: ', \
        ##      self.new_modes)

    #@property
    #def memory_modes(self):
    #    """ GW memory SPH modes """
    #    if self._memory_modes is None:
    #        self._memory_modes = {kk: vv * self.amp_rescale \
    #                              for kk, vv in self.h_mem_sxs.items()}
    #    return self._memory_modes

    #def _calculate_memory(self):
    #    """ Calculation is performed in SXS in dimensionless units """

    #    # In an old way, based on gwtools.sxs_memory:
    #    #self.h_mem_sxs, self.times_sxs = sxs_memory(self._lal_hlms_du, \
    #    #                                            self._lal_time_hlms_du)

    @property
    def calculate_J_E_modes(self):
        return True if 'J_E' in self.data_params['memory_contributions'] else False

    @property
    def J_E_modes(self):
        """ Displacement memory modes, dimensionless units """
        if self._J_E_modes is None:
            ref_time = time.time()
            self._J_E_modes = sxs_waveform_to_h_lm_dict(\
                        sxs.waveforms.memory.J_E(self.sxs_hlms_du), \
                        self.possible_modes)
            print('[!] Time for J_E calculation: ', time.time()-ref_time)
        return self._J_E_modes

    @property
    def J_E(self):
        return self._J_E

    @J_E.setter
    def J_E(self, new_J_E):
        self._J_E = new_J_E

    @property
    def J_J(self):
        return self._J_J

    @J_J.setter
    def J_J(self, new_J_J):
        self._J_J = new_J_J

    @property
    def timevector(self):
        return self.time_array_for_lal_timeseries(self._lal_hlms[(2,2)])

    @property
    def J_E_tilde(self):
        return (fft_wrapper(self.J_E[0], self.timevector, self.delta_t, self.f_min, self.f_max, self.gw_params['geocent_time'])[0], fft_wrapper(self.J_E[1], self.timevector, self.delta_t, self.f_min, self.f_max, self.gw_params['geocent_time'])[0])

    @property
    def J_J_tilde(self):
        return (fft_wrapper(self.J_J[0], self.timevector, self.delta_t, self.f_min, self.f_max, self.gw_params['geocent_time'])[0], fft_wrapper(self.J_J[1], self.timevector, self.delta_t, self.f_min, self.f_max, self.gw_params['geocent_time'])[0])

    @property
    def calculate_J_J_modes(self):
        return True if 'J_J' in self.data_params['memory_contributions'] else False

    @property
    def J_J_modes(self):
        """ Flux part of the spin memory modes, dimensionless units """
        if self._J_J_modes is None:
            ref_time = time.time()
            self._J_J_modes = sxs_waveform_to_h_lm_dict(\
                        sxs.waveforms.memory.J_J(self.sxs_hlms_du), \
                        self.possible_modes)
            print('[!] Time for J_J calculation: ', time.time()-ref_time)
        return self._J_J_modes

    @property
    def _lal_time_hlms_du(self):
        """ In dimensionless units (du) """
        return self.time_array_for_lal_timeseries(self._lal_hlms[(2,2)]) / self.t_rescale

    @property
    def _lal_hlms_du(self):
        """ In dimensionless units (du) """
        return {kk: vv.data.data / self.amp_rescale \
                for kk, vv in self._lal_hlms.items()}

    @property
    def max_ll_lal_hlms(self):
        return max(self._lal_hlms.keys())[0]

    @property
    def min_ll_lal_hlms(self):
        return min(self._lal_hlms.keys())[0]

    @property
    def possible_modes(self):
        """ All possible SPH modes based on l_max found in LAL output """
        return [(ll, mm) for ll in range(2, self.max_ll_lal_hlms+1) for mm in range(-ll,ll+1)]

    def _add_missing_lms_in_lal_hlms(self):
        _lm = (2,2)
        for (ll,mm) in self.possible_modes:
            if (ll,mm) not in self._lal_hlms.keys():
                _temp_lal_hlms_ll_mm = lal.CreateCOMPLEX16TimeSeries(
                                          str((ll,mm))+' mode',
                                          self._lal_hlms[_lm].epoch,
                                          self._lal_hlms[_lm].f0,
                                          self._lal_hlms[_lm].deltaT,
                                          self._lal_hlms[_lm].sampleUnits,
                                          self._lal_hlms[_lm].data.length)
                if (ll,-mm) in self._lal_hlms.keys() and mm<0:
                    # Faking negative modes
                    _temp_lal_hlms_ll_mm.data.data = (-1)**(ll) * np.conjugate(self._lal_hlms[(ll,-mm)].data.data)
                else:
                    _temp_lal_hlms_ll_mm.data.data = np.zeros( self._lal_hlms[_lm].data.length )
                self._lal_hlms[(ll,mm)] = _temp_lal_hlms_ll_mm

    @property
    def sxs_hlms_du(self):
        """
        Strain SPH modes converted to SXS format. In dimensionless units (du), 
        all possible modes.

        Update: negative modes need to be set to zeros for memory calculation?
        Otherwise, the amplitude becomes higher than expected.
        """
        if self._sxs_hlms_du is None:
            out = []
            for (ll,mm) in self.possible_modes:
                if (ll,mm) in self._lal_hlms.keys():
                    ## Updaet: m<0 modes are necessary
                    #if mm>=0 or self.data_params['mem_neg_modes']:
                    out.append(self._lal_hlms[(ll,mm)].data.data / \
                               self.amp_rescale)
                    #else:
                    #  out.append(np.zeros(self._lal_hlms[(ll,mm)].data.length))
                else:
                    raise ValueError('This should not happen, all cases are handled before in _add_missing_lms_in_lal_hlms')
                #elif (ll,-mm) in self._lal_hlms.keys() and mm<0:
                #    # Faking negative modes
                #    out.append((-1)**(ll) * np.conjugate(\
                #                            self._lal_hlms[(ll,mm)].data.data / \
                #                            self.amp_rescale))
                #else:
                #    out.append( np.zeros(self._lal_hlms[(2,2)].data.length) )

            self._sxs_hlms_du =  sxs.waveforms.WaveformModes(np.array(out).T, 
                                               time=self._lal_time_hlms_du, 
                                               modes_axis=1, time_axis=0, 
                                               ell_min=self.min_ll_lal_hlms, 
                                               ell_max=self.max_ll_lal_hlms)
            self._sxs_hlms_du._metadata['spin_weight'] = -2
        return self._sxs_hlms_du

class NRSurSPH_Memory(wf.Waveform):
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self.reset_waveform_evaluation()
        self.timevector = None
        self.timevector_du = None
        self._time_domain_f_min = None
        self._import_nrsurrogate()

        #self.calculate_time_domain_sph_du() # td
        #self._add_missing_lms_in_lal_hlms() # td
        #self.sxs_hlms_du # Td
        #self._add_memory() # td 
        #self._td_strain_from_sph_du() # td 
        #import ipdb; ipdb.set_trace()

    def reset_waveform_evaluation(self):
        """ Clear all cache (imporant) """
        # Time-domain specific
        self._m1_m2_to_mtot_q()
        self._init_lal_gw_parameters()
        self._sxs_hlms_du = None
        self._ht_plus = None
        self._ht_cross = None
        # Memory-specific
        self._fd_memory = 0.
        self._J_J_modes = None
        self._J_E_modes = None

    def update_gw_params(self, new_gw_params):
        self._update_gw_params_common(new_gw_params)
        self.reset_waveform_evaluation()

    def _m1_m2_to_mtot_q(self):
        self.gw_params['total_mass'] = (self.gw_params['mass_1'] + self.gw_params['mass_2']) * (1 + self.gw_params['redshift'])
        self.gw_params['mass_ratio'] = self.gw_params['mass_1'] / self.gw_params['mass_2']
        if self.gw_params['mass_1'] < self.gw_params['mass_2']:
            self.gw_params['mass_ratio'] = 1. / self.gw_params['mass_ratio']

    @property
    def time_domain_f_min(self):
        """
        In gwsurrogate, f_min for a given waveform has to correspond to earlier 
        or equal time to the minimum time of a given timevector (which we fix 
        for the first waveform evaluation for a waveform object). In Fisher 
        matrix calculation, perturbing distance/mass might affect start time 
        of the waveform. So, time-domain f_min has to be slightly lower to 
        avoid error. Moreover, frequencyvector (and, thus, f_min) is 
        overwritten here.
        """
        if self._time_domain_f_min is None:
            if 'time_domain_f_min' in self.data_params:
                self._time_domain_f_min = self.data_params['time_domain_f_min']
            else:
                self._time_domain_f_min = self.f_min
                logging.warning('Setting f_min to {}'.format(\
                                self._time_domain_f_min))
        return self._time_domain_f_min

    @property
    def _gw_params_for_spin_conversion(self):
        return ['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2',
                'phi_12', 'a_1', 'a_2', 'mass_1', 'mass_2', 'phase']

    def _init_lal_gw_parameters(self):
        gwfish_input_params = {kk: self.gw_params[kk] for kk in self._gw_params_for_spin_conversion}
        self.gw_params['iota'], self.gw_params['spin_1x'], \
            self.gw_params['spin_1y'], self.gw_params['spin_1z'], \
            self.gw_params['spin_2x'], self.gw_params['spin_2y'], \
            self.gw_params['spin_2z'] = wf.bilby_to_lalsimulation_spins(\
            reference_frequency=self.f_ref, **gwfish_input_params)

    def f_to_nat(self, ff):
        """ Frequency to natural units """
        return ff * self.t_rescale

    def t_to_nat(self, tt):
        """ Time to natural units """
        return tt / self.t_rescale

    @property
    def amp_rescale(self):
        """
        A multiplicative coefficient to convert GW time series data to/from
        dimensionless units.
        """
        return amp_rescale_coeff(self.gw_params['luminosity_distance'], \
                                 self.gw_params['total_mass'])

    @property
    def t_rescale(self):
        """
        A multiplicative coefficient to convert time array corresponding to
        GW time series data to/from dimensionless units.
        """
        return time_rescale_coeff(self.gw_params['total_mass'])

    @property
    def l_max(self):
        if self._l_max is None:
            if 'l_max' in self.data_params:
                self.l_max = self.data_params['l_max']
            else:
                self.l_max = 4
                logging.warning('Setting l_max to {}'.format(self._l_max))
        return self._l_max

    @property
    def max_ll_hlms(self):
        return max(self._time_domain_sph_du.keys())[0]

    @property
    def min_ll_hlms(self):
        return min(self._time_domain_sph_du.keys())[0]

    @property
    def possible_modes(self):
        """ All possible SPH modes based on l_max found in LAL output """
        return [(ll, mm) for ll in range(2, self.max_ll_hlms+1) for mm in range(-ll,ll+1)]

    def calculate_time_domain_strain(self):
        self.calculate_time_domain_sph_du() # td
        self._add_missing_lms_in_lal_hlms() # td
        self.sxs_hlms_du # Td
        self._add_memory() # td
        self._td_strain_from_sph_du()
        self.timevector = self.timevector_du * self.t_rescale
        # Tapering
        self._ht_plus = self._taper_timeseries_lal(self._ht_plus)
        self._ht_cross = self._taper_timeseries_lal(self._ht_cross)
        #self._waveform_postprocessing()
        htp = self._ht_plus[:, np.newaxis]
        htc = self._ht_cross[:, np.newaxis]
        polarizations = np.hstack((htp, htc))
        self._time_domain_strain = polarizations

    @property
    def _lal_mass_1(self):
        return self.gw_params['mass_1'] * lal.MSUN_SI * (1 + self.gw_params['redshift'])

    @property
    def _lal_mass_2(self):
        return self.gw_params['mass_2'] * lal.MSUN_SI * (1 + self.gw_params['redshift'])

    @property
    def fisco(self):
        return 1.0 / (np.power(6.0, 1.5) * lal.PI * \
               (self._lal_mass_1 + self._lal_mass_2) * \
               lal.MTSUN_SI / lal.MSUN_SI)

    def _taper_timeseries_lal(self, timeseries):
        """
        Data conditioning: option 1, that might change fRef.
        This is based on SimInspiralTDfromTD.
        """

        extra_time_fraction = 0.1
        extra_cycles = 3.0
        textra = extra_cycles / self.f_min

        tchirp = lalsim.SimInspiralChirpTimeBound(self.f_min,
            self._lal_mass_1,
            self._lal_mass_2,
            self.gw_params['spin_1z'], self.gw_params['spin_2z'])

        # Tapering beginning of the waveform, several cycles
        timeseries = taper_1(timeseries, self.delta_t, extra_time_fraction * tchirp + textra)
        # Tapering one cycle at f_min at the beginning and one cycle at f_isco
        # at the end
        timeseries = taper_2(timeseries, self.delta_t, self.f_min, self.fisco)
        return timeseries

    def _fd_gwfish_output_format(self, hfp, hfc):

        hfp = hfp[:, np.newaxis]
        hfc = hfc[:, np.newaxis]

        polarizations = np.hstack((hfp, hfc))

        return polarizations

    def calculate_frequency_domain_strain(self):
        if self.time_domain_strain is None:
            self.calculate_time_domain_strain()

        # Zero pad data, to make it 2**N length
        #zp_length = int(2**np.ceil(np.log2(len(self._ht_plus))))
        #extra_length = zp_length - len(self._ht_plus)
        #pad_1 = extra_length // 2
        #pad_2 = pad_1 + extra_length % 2
        #ht_plus_zp = np.concatenate([np.zeros(pad_1),self._ht_plus,np.zeros(pad_2)])
        #ht_cross_zp = np.concatenate([np.zeros(pad_1),self._ht_cross,np.zeros(pad_2)])
        #times_add_1 = np.arange(self.timevector[0]-self.delta_t*pad_1,self.timevector[0],self.delta_t)
        #times_add_2 = np.arange(self.timevector[-1]+self.delta_t,self.timevector[-1]+self.delta_t*(pad_2+1),self.delta_t)
        #timevector_zp = np.concatenate([times_add_1,self.timevector,times_add_2])

        # FFT
        self._hf_plus, ff = fft_np(self._ht_plus, self.delta_t, self.timevector[0], self.timevector[-1])
        self._hf_cross, ff = fft_np(self._ht_cross, self.delta_t, self.timevector[0], self.timevector[-1])
        mask_f = (ff >= self.f_min) * (ff <= self.f_max)
        self.frequencyvector = ff[mask_f]
        self._hf_plus = self._hf_plus[mask_f]
        self._hf_cross = self._hf_cross[mask_f]


        # Phase correction by epoch and delta t
        dt = 1/self.delta_f + self.timevector[0]
        self._hf_plus *= np.exp(-1j * 2 * np.pi * dt * self.frequencyvector)
        self._hf_cross *= np.exp(-1j * 2 * np.pi * dt * self.frequencyvector)

        # Phase correction by geocentric time
        phi_in = np.exp(1.j*(2*self.frequencyvector*np.pi*self.gw_params['geocent_time']))
        self._hf_plus = phi_in * np.conjugate(self._hf_plus)
        self._hf_cross = phi_in * np.conjugate(self._hf_cross)
        polarizations = self._fd_gwfish_output_format(self._hf_plus, self._hf_cross)
        self._frequency_domain_strain = polarizations

        # Create LAL timeseries
        #self._lal_ht_plus = lal.CreateREAL8TimeSeries('h_plus', self.gw_params['geocent_time'], self.f_min, self.delta_t, 'strain', zp_length)
        #self._lal_ht_cross = lal.CreateREAL8TimeSeries('h_cross', self.gw_params['geocent_time'], self.f_min, self.delta_t, 'strain', zp_length)
        #self._lal_ht_plus.epoch = self.gw_params['geocent_time']
        #self._lal_ht_cross.epoch = self.gw_params['geocent_time']
        #self._lal_ht_plus.data.data = ht_plus_zp
        #self._lal_ht_cross.data.data = ht_cross_zp
        ##self._lal_ht_plus = lal.CreateREAL8TimeSeries('h_plus', self.gw_params['geocent_time'], self.f_min, self.delta_t, 'strain', len(self.timevector))
        ##self._lal_ht_cross = lal.CreateREAL8TimeSeries('h_cross', self.gw_params['geocent_time'], self.f_min, self.delta_t, 'strain', len(self.timevector))
        ##self._lal_ht_plus.epoch = self.gw_params['geocent_time']
        ##self._lal_ht_cross.epoch = self.gw_params['geocent_time']
        ##self._lal_ht_plus.data.data = self._ht_plus
        ##self._lal_ht_cross.data.data = self._ht_cross

        ##self._lal_ht_plus.data.data[] = self._ht_plus
        ###self._lal_ht_cross.data.data[] = self._ht_cross

        ## FFT
        #lal_hf_plus = fft.fft_lal_timeseries(self._lal_ht_plus, self.delta_f)
        #lal_hf_cross = fft.fft_lal_timeseries(self._lal_ht_cross, self.delta_f)
        #lal_ff = np.arange(lal_hf_plus.f0, lal_hf_plus.data.length*lal_hf_plus.deltaF, lal_hhf_plus.deltaF)
        #mask_ff = (lal_ff >= self.f_min) * (lal_ff <= self.f_max)
        #self.frequencyvector = lal_ff[mask_ff]
        #self._hf_plus = lal_hf_plus.data.data[mask_ff]
        #self._hf_cross = lal_hf_cross.data.data[mask_ff]

    @property
    def timevector_du(self):
        return self._timevector_du

    @timevector_du.setter
    def timevector_du(self, newtimevector_du):
        self._timevector_du = newtimevector_du

    def calculate_time_domain_sph_du(self):
        if self.timevector_du is None:
            self.timevector_du, self._time_domain_sph_du, self.dyn = \
                self.sur(self.gw_params['mass_ratio'], \
                [self.gw_params['spin_1x'], \
                self.gw_params['spin_1y'], \
                self.gw_params['spin_1z']], \
                [self.gw_params['spin_2x'], \
                self.gw_params['spin_2y'], \
                self.gw_params['spin_2z']], \
                dt=self.t_to_nat(self.delta_t), \
                f_low=self.f_to_nat(self.time_domain_f_min), \
                f_ref=self.f_to_nat(self.f_ref), ellMax=4, \
                #M=self.gw_params['total_mass'], \
                #dist_mpc=self.gw_params['luminosity_distance'], \
                #inclination=self.gw_params['iota'], \
                phi_ref=0., units='dimensionless')
        else:
            _, self._time_domain_sph_du, self.dyn = \
                self.sur(self.gw_params['mass_ratio'], \
                [self.gw_params['spin_1x'], \
                self.gw_params['spin_1y'], \
                self.gw_params['spin_1z']], \
                [self.gw_params['spin_2x'], \
                self.gw_params['spin_2y'], \
                self.gw_params['spin_2z']], \
                times=self.timevector_du, \
                f_low=self.f_to_nat(self.time_domain_f_min), \
                f_ref=self.f_to_nat(self.f_ref), ellMax=4, \
                #M=self.gw_params['total_mass'], \
                #dist_mpc=self.gw_params['luminosity_distance'], \
                #inclination=self.gw_params['iota'], \
                phi_ref=0., units='dimensionless')

    @property
    def sxs_hlms_du(self):
        """ Strain SPH modes converted to SXS format. In dimensionless units (du), all possible modes. """
        if self._sxs_hlms_du is None:
            out = [self._time_domain_sph_du[(ll,mm)] for (ll,mm) in self.possible_modes]

            self._sxs_hlms_du =  sxs.waveforms.WaveformModes(np.array(out).T,
                                               time=self.timevector_du,
                                               modes_axis=1, time_axis=0,
                                               ell_min=self.min_ll_hlms,
                                               ell_max=self.max_ll_hlms)
            self._sxs_hlms_du._metadata['spin_weight'] = -2
        return self._sxs_hlms_du


    def _add_missing_lms_in_lal_hlms(self):
        _lm = (2,2)
        for (ll,mm) in self.possible_modes:
            if (ll,mm) not in self._time_domain_sph_du.keys():
                if (ll,-mm) in self._time_domain_sph_du.keys() and mm<0:
                    # Faking negative modes
                    self._time_domain_sph_du[(ll,mm)] = (-1)**(ll) * np.conjugate(self._time_domain_sph_du[(ll,-mm)])
                else:
                    self._time_domain_sph_du[(ll,mm)] = np.zeros( len(self._time_domain_sph_du[_lm]), dtype=np.complex128)

    def _td_strain_from_sph_du(self):
        _m = (2,2)
        self._ht_plus = 0.
        self._ht_cross = 0.
        fake_neg_modes = not np.any([mm < 0 for (ll, mm) in self._time_domain_sph_du])

        for (ll, mm) in self._time_domain_sph_du:

            ylm = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], # inclination
                                                    self.gw_params['phase'], -2, ll, mm)
            # LAL: Cross-polarization is the *negative* of the imaginary part
            self._ht_plus += np.real(ylm * self._time_domain_sph_du[(ll, mm)] * self.amp_rescale)
            self._ht_cross -= np.imag(ylm * self._time_domain_sph_du[(ll, mm)] * self.amp_rescale)
            # If m<0 modes are not in the dictionary of modes, we calculate
            # them from m>0 modes
            if fake_neg_modes and mm>0:
                print('BG: faking negative modes from SPH du')
                yl_m = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'],
                                                        self.gw_params['phase'], -2, ll, -mm)
                self._ht_plus += np.real(yl_m * (-1)**(ll) * np.conjugate(self._time_domain_sph_du[(ll, mm)]) * self.amp_rescale)
                self._ht_cross -= np.imag(yl_m * (-1)**(ll) * np.conjugate(self._time_domain_sph_du[(ll, mm)]) * self.amp_rescale)
        # Here, waveform is not conditioned

    def _add_memory(self):
        if self.calculate_J_J_modes:
            for lm in self.possible_modes:
                self._time_domain_sph_du[lm] += self.J_J_modes[lm] * self.gw_params['J_J']
        if self.calculate_J_E_modes:
            for lm in self.possible_modes:
                self._time_domain_sph_du[lm] += self.J_E_modes[lm] * self.gw_params['J_E']

    @property
    def calculate_J_E_modes(self):
        return True if 'J_E' in self.data_params['memory_contributions'] else False

    @property
    def J_E_modes(self):
        """ Displacement memory modes, dimensionless units """
        if self._J_E_modes is None:
            ref_time = time.time()
            self._J_E_modes = sxs_waveform_to_h_lm_dict(\
                        sxs.waveforms.memory.J_E(self.sxs_hlms_du), \
                        self.possible_modes)
            print('[!] Time for J_E calculation: ', time.time()-ref_time)
        return self._J_E_modes

    @property
    def J_E(self):
        _len = len(self.J_E[(2,2)])
        J_E_plus, J_E_cross = np.zeros(_len), np.zeros(_len)
        for (ll, mm) in self.J_E_modes:
            # To test modes, not a good idea to use because of possible confusion
            #if ll > self.l_max:
            #    continue
            ylm = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'], # inclination
                                                    self.gw_params['phase'], -2, ll, mm)
            # LAL: Cross-polarization is the *negative* of the imaginary part
            J_E_plus += np.real(ylm * self.J_E_modes[(ll, mm)].data.data)
            J_E_cross -= np.imag(ylm * self.J_E_modes[(ll, mm)].data.data)

        return J_E_plus, J_E_cross

    @property
    def calculate_J_J_modes(self):
        return True if 'J_J' in self.data_params['memory_contributions'] else False

    @property
    def J_J_modes(self):
        """ Flux part of the spin memory modes, dimensionless units """
        if self._J_J_modes is None:
            ref_time = time.time()
            self._J_J_modes = sxs_waveform_to_h_lm_dict(\
                        sxs.waveforms.memory.J_J(self.sxs_hlms_du), \
                        self.possible_modes)
            print('[!] Time for J_J calculation: ', time.time()-ref_time)
        return self._J_J_modes

    # Bellow is for pure time-domain class, no SPH
    #def calculate_time_domain_strain(self):
    #    self.timevector, self._time_domain_strain, self.dyn = \
    #        self.sur(self.gw_params['mass_ratio'], \
    #        [self.gw_params['spin_1x'], \
    #        self.gw_params['spin_1y'], \
    #        self.gw_params['spin_1z']], \
    #        [self.gw_params['spin_2x'], \
    #        self.gw_params['spin_2y'], \
    #        self.gw_params['spin_2z']], \
    #        dt=self.delta_t, f_low=self.f_min, f_ref=self.f_ref, ellMax=4, \
    #        M=self.gw_params['total_mass'], \
    #        dist_mpc=self.gw_params['luminosity_distance'], \
    #        inclination=self.gw_params['iota'], \
    #        phi_ref=0., units='mks')

    def _import_nrsurrogate(self):
        try:
            import gwsurrogate
        except:
            raise ValueError('Module gwsurrogate not found.')
        env_vars = os.environ
        if 'LAL_DATA_PATH' in env_vars:
            if self.name+'.h5' in os.listdir(os.environ['LAL_DATA_PATH']):
                self.sur = gwsurrogate.LoadSurrogate(os.environ['LAL_DATA_PATH']+self.name+'.h5')
            else:
                err_message = 'File '+self.name+'.h5'+' mus be in '+os.environ['LAL_DATA_PATH']
                raise ValueError(err_message)
        else:
            raise ValueError('Please set LAL_DATA_PATH and put surrogate waveform files there.')

    def clear_nrsurrogate(self):
        """ Needed before pickling """
        self.sur = None


    #def calculate_time_domain_strain(self):
    #    tt, hh, dyn = sur(self.gw_params['mass_ratio'], 
    #                      [self.gw_params['spin_1x'], self.gw_params['spin_1y'], self.gw_params['spin_1z']],
    #                      [self.gw_params['spin_2x'], self.gw_params['spin_2y'], self.gw_params['spin_2z']], 
    #                      dt=self.delta_t, f_low=3., f_ref=self.f_ref, ellMax=4., M=M_tot, dist_mpc=luminosity_distance, inclination=theta_jn, phi_ref=phase, units='mks')

    #    self._waveform_postprocessing()

    #    htp = self.ht_plus_out[:, np.newaxis]
    #    htc = self.ht_cross_out[:, np.newaxis]

    #    polarizations = np.hstack((htp, htc))

    #    print('Warning: inverting hx in LAL caller')

    #    self._time_domain_strain = polarizations



