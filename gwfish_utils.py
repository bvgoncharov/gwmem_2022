import copy
import logging

import numpy as np

import lal
import lalsimulation as lalsim

import GWFish.modules.waveforms as wf
import GWFish.modules.fft as fft

import gwtools
from gwtools import sxs_memory
import sxs

import time

try:
    import gwsurrogate
    from scipy.signal.windows import tukey
    from scipy import interpolate
    #sur = gwsurrogate.LoadSurrogate('NRSur7dq4')
    sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

    import copy # only temporary
except ModuleNotFoundError as err_gwsur:
    print('Module gwsurrogate not found. Surrogate waveforms are not available.')

# FOR DEBUGGING
from matplotlib import pyplot as plt

#def hphc_amplitudes(waveform, parameters, frequencyvector, time_domain=False, sph_modes = False, plot=None):
#    parameters = parameters.copy()
#
#    if waveform == 'gwfish_TaylorF2':
#        hphc = wf.TaylorF2(parameters, frequencyvector, plot=plot)
#    elif waveform == 'gwfish_IMRPhenomD':
#        hphc = wf.IMRPhenomD(parameters, frequencyvector, plot=plot)
#    elif waveform[0:7] == 'lalsim_':
#        if time_domain:
#            data_params = {'frequencyvector': frequencyvector}
#            if sph_modes:
#                waveform_obj = LALTD_SPH_Waveform(waveform[7:], parameters, data_params)
#            else:
#                waveform_obj = wf.LALTD_Waveform(waveform[7:], parameters, data_params)
#            #waveform_obj.calculate_sph_modes()
#
#            hphc = waveform_obj()
#            #hp_lal = waveform_obj._lal_ht_plus
#            #hc_lal = waveform_obj._lal_ht_cross
#        else:
#            data_params = {'frequencyvector': frequencyvector}
#            waveform_obj = wf.LALFD_Waveform(waveform[7:], parameters, data_params)
#            hphc = waveform_obj()
#    elif waveform[0:6] == 'nrsur_':
#        hphc, frequencyvector = nrsur_caller(waveform[6:], frequencyvector, **parameters)
#    elif waveform[0:6] == 'memes_':
#        hphc, frequencyvector = nrsur_memestr(waveform[6:], frequencyvector, **parameters)
#    else:
#        waveform_error = '{} is not a valid waveform. '.format(str(waveform)) + \
#                         'Valid options are gwfish_TaylorF2, gwfish_IMRPhenomD, lalsim_XXX.'
#        raise ValueError(waveform_error)
#
#    if time_domain and not waveform[0:6] == 'nrsur_':
#        # Here it is not really t_of_f, it is just a time vector
#        t_of_f = np.arange(0,hphc.shape[0]*waveform_obj.delta_t,waveform_obj.delta_t)
#        t_of_f = np.expand_dims(t_of_f, axis=1)
#    else:
#        t_of_f = wf.t_of_f_PN(parameters, frequencyvector)
#
#        if fmax := parameters.get('max_frequency', None):
#            for i in range(2):
#                hphc[:, i] = np.where(frequencyvector[:, 0] <= fmax, hphc[:, i], 0j)
#
#    if not waveform[0:6] == 'nrsur_' and len(frequencyvector) != len(waveform_obj.frequencyvector):
#        frequencyvector = np.expand_dims(waveform_obj.frequencyvector,axis=1)
#
#    return hphc, t_of_f, frequencyvector

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

class LALTD_SPH_Waveform(wf.LALTD_Waveform):
    """
    Calls SPH modes from LAL in time domain, sums modes, computes frequency-domain waveform.
    """
    def __init__(self, name, gw_params, data_params):
        self.l_max = None
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
        return 3.

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
            self._lalsim_args = [
                0, # phiRef, unused parameter
                self.delta_t,
                self._lal_mass_1,
                self._lal_mass_2,
                self.gw_params['spin_1x'], self.gw_params['spin_1y'], self.gw_params['spin_1z'],
                self.gw_params['spin_2x'], self.gw_params['spin_2y'], self.gw_params['spin_2z'],
                self.time_domain_f_min,
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
                yl_m = lal.SpinWeightedSphericalHarmonic(self.gw_params['iota'],
                                                        self.gw_params['phase'], -2, ll, -mm)
                self._lal_ht_plus.data.data += np.real(yl_m * (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data))
                self._lal_ht_cross.data.data -= np.imag(yl_m * (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data))
        # Here, waveform is not conditioned

    def calculate_time_domain_strain(self):
        self.calculate_time_domain_sph_modes()
        self._td_strain_from_sph_modes()
        self._waveform_postprocessing()

    def _taper(self):
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

        # Calling data conditioning
        lalsim.SimInspiralTDConditionStage1(self._lal_ht_plus, self._lal_ht_cross, extra_time_fraction * tchirp + textra, self.time_domain_f_min)
        lalsim.SimInspiralTDConditionStage2(self._lal_ht_plus, self._lal_ht_cross, self.f_min, fisco)

    def _ht_postproccessing_SimInspiralCTDM(self):
        self._taper()
        self._ht_postproccessing_SimInspiralTD()


class LALTD_SPH_Memory(LALTD_SPH_Waveform):
    def __init__(self, name, gw_params, data_params):
        self.l_max = None
        super(LALTD_SPH_Waveform, self).__init__(name, gw_params, data_params)
        self._sxs_hlms_du = None
        self._J_J_modes = None
        self._J_E_modes = None

    def update_gw_params(self, new_gw_params):
        self.gw_params.update(new_gw_params)
        self._frequency_domain_strain = None
        self._time_domain_strain = None
        # Specific to LALTD_SPH_Memory (this class)
        #self._J_J_modes = None
        #self._J_E_modes = None
        # Specific to LALFD_Waveform
        self._init_lambda()
        self._init_lal_gw_parameters()
        self._setup_lal_caller_args()

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
                                 self.m_tot)

    @property
    def t_rescale(self):
        """
        A multiplicative coefficient to convert time array corresponding to 
        GW time series data to/from dimensionless units.
        """
        return time_rescale_coeff(self.m_tot)

    @property
    def td_memory(self):
        return None

    @property
    def fd_memory(self):
        return None

    #def _fake_negative_sph_modes(self):
    #    fake_neg_modes = not np.any([mm < 0 for (ll, mm) in self._lal_hlms])
    #    for (ll, mm) in self._lal_hlms:

    #    (-1)**(ll) * np.conjugate(self._lal_hlms[(ll, mm)].data.data)

    def _add_memory(self):
        if self.calculate_J_J_modes:
            for lm in self.possible_modes:
                self._lal_hlms[lm].data.data += self.J_J_modes[lm] * self.amp_rescale
        if self.calculate_J_E_modes:
            for lm in self.possible_modes:
                self._lal_hlms[lm].data.data += self.J_E_modes[lm] * self.amp_rescale

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
        """ Strain SPH modes converted to SXS format. In dimensionless units (du), all possible modes. """
        if self._sxs_hlms_du is None:
            out = []
            for (ll,mm) in self.possible_modes:
                if (ll,mm) in self._lal_hlms.keys():
                    out.append(self._lal_hlms[(ll,mm)].data.data / \
                               self.amp_rescale)
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

def nrsur_memestr(waveform, frequencyvector, mass_1, mass_2, luminosity_distance, redshift, theta_jn, phase, geocent_time,
           a_1=0, tilt_1=0, phi_12=0, a_2=0, tilt_2=0, phi_jl=0, lambda_1=0, lambda_2=0, **kwargs):

    # Conversion
    if mass_1 < mass_2:
      mass_x = copy.copy(mass_1)
      mass_1 = copy.copy(mass_2)
      mass_2 = mass_x
      # Above quick fix
      #raise ValueError('Must be mass_1 >= mass_2')
    qq = mass_1/mass_2                # q = m1/m2 >= 1
    M_tot = (mass_1 + mass_2)*(1+redshift)
    f_ref = frequencyvector[0,0]  # BORIS: was 50 for LAL # Reference frequecny in Hz. The spins are assumed to specified at this frequency
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = wf.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=f_ref, phase=phase)

    chiA = [spin_1x, spin_1y, spin_1z]          # dimensionless spin of the heavier BH
    chiB = [spin_2x, spin_2y, spin_2z]         # dimensionless spin of the lighter BH

    #f_low = 0                        # initial frequency, f_low=0 returns the full surrogate

    ellMax = 4                       # Highest ell index for modes to use

    # dyn stands for dynamics, do dyn.keys() to see contents
    if sur._domain_type == 'Time':
        dt = 0.5/frequencyvector[-1,0] #1./4096                     # step size in seconds


        f_max = frequencyvector[-1,0]
        delta_f = frequencyvector[1,0]-frequencyvector[0,0]
        t_obs_nrsur = (2 * f_max * dt) / delta_f
        # Optimally, here we should calculate f_gw(-t_obs_nrsur)
        #nrsur_f_low = np.min(frequencyvector)
        nrsur_f_low = 3.
        tt, hh, dyn = sur(qq, chiA, chiB, dt=dt, f_low=nrsur_f_low, f_ref=f_ref, ellMax=ellMax, M=M_tot, dist_mpc=luminosity_distance, inclination=theta_jn, phi_ref=phase, units='mks')

        import memestr
        memestr.waveforms.nrhybsur3dq8.fd_nr_sur
        memestr.waveforms.nrhybsur3dq8.fd_nr_sur(np.squeeze(frequencyvector),qq,M_tot,chiA,chiB,luminosity_distance,theta_jn,phase)
        import ipdb; ipdb.set_trace()

class GWSurrogate_Waveform(wf.LALTD_Waveform):
    def __init__(self, name, gw_params, data_params):
        super().__init__(name, gw_params, data_params)
        self._m1_m2_to_mtot_q()

    def _m1_m2_to_mtot_q(self):
        self.gw_params['total_mass'] = self.gw_params['mass_1'] + self.gw_params['mass_2']
        self.gw_params['mass_ratio'] = self.gw_params['mass_1'] / self.gw_params['mass_2']
        if self.gw_params['mass_1'] < self.gw_params['mass_2']:
            self.gw_params['mass_ratio'] = 1. / self.gw_params['mass_ratio']

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

def nrsur_caller(waveform, frequencyvector, mass_1, mass_2, luminosity_distance, redshift, theta_jn, phase, geocent_time,
           a_1=0, tilt_1=0, phi_12=0, a_2=0, tilt_2=0, phi_jl=0, lambda_1=0, lambda_2=0, **kwargs):
    """
    kwargs:
    - fft_roll_off: roll-off for the FFT window in seconds. Zero-padding data will be added after the end of the waveform,
    so that windowing does not apply to the merger part of the signal.
    """

    # Conversion
    if mass_1 < mass_2:
      mass_x = copy.copy(mass_1)
      mass_1 = copy.copy(mass_2)
      mass_2 = mass_x
      # Above quick fix
      #raise ValueError('Must be mass_1 >= mass_2')
    qq = mass_1/mass_2                # q = m1/m2 >= 1
    M_tot = (mass_1 + mass_2)*(1+redshift)
    f_ref = frequencyvector[0,0]  # BORIS: was 50 for LAL # Reference frequecny in Hz. The spins are assumed to specified at this frequency
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = wf.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=f_ref, phase=phase)

    chiA = [spin_1x, spin_1y, spin_1z]          # dimensionless spin of the heavier BH
    chiB = [spin_2x, spin_2y, spin_2z]         # dimensionless spin of the lighter BH

    #f_low = 0                        # initial frequency, f_low=0 returns the full surrogate

    ellMax = 4                       # Highest ell index for modes to use

    # dyn stands for dynamics, do dyn.keys() to see contents
    if sur._domain_type == 'Time':
        dt = 0.5/frequencyvector[-1,0] #1./4096                     # step size in seconds

        ## For adding memory (TO-DO)
        ## Ok, what I need to do:
        ## 1. Convert time/frequency parameters to natural units t[M]
        ## 2. Call nrsur in natural units, without M_tot, D_L, and without inclination to get modes
        ## 3. Add memory
        ## 4. Convert back to observable units (with M_tot, D_L) and sum modes (with inclination)
        ## 5. Condition the waveform and FFT

        #tt, hh, dyn = sur(qq, chiA, chiB, dt=0.1, f_low=0.02, ellMax=4)#np.min(frequencyvector), f_ref=f_ref)

        ## Test
        #from gwtools import sxs_memory
        #from matplotlib import pyplot as plt
        #import tqdm


        #h_mem_sxs, times_sxs = sxs_memory(hh, tt)
        #plt.close()
        #for kk in tqdm.tqdm(hh.keys()):
        #    plt.plot(times_sxs, np.real(h_mem_sxs[kk]),label='After memory '+str(kk))
        #    plt.plot(tt, np.real(hh[kk]), label='Before memory '+str(kk))
        #    plt.xlabel('t')
        #    plt.ylabel('h')
        #    plt.legend()
        #    plt.savefig('/Users/boris.goncharov/projects/out_gwmem_2022/mem_modes_test/nrsur_'+str(kk)+'.png')
        #    plt.close()


        # f_low=np.min(frequencyvector) (2 Hz) yields an error related to omega for NRSur2, but not for NRSur4
        # For NRHybSur3dq8, any minimum frequency works (including 2 Hz for ET)
        f_max = frequencyvector[-1,0]
        delta_f = frequencyvector[1,0]-frequencyvector[0,0]
        t_obs_nrsur = (2 * f_max * dt) / delta_f
        # Optimally, here we should calculate f_gw(-t_obs_nrsur)
        #nrsur_f_low = np.min(frequencyvector)
        nrsur_f_low = 3.
        tt, hh, dyn = sur(qq, chiA, chiB, dt=dt, f_low=nrsur_f_low, f_ref=f_ref, ellMax=ellMax, M=M_tot, dist_mpc=luminosity_distance, inclination=theta_jn, phi_ref=phase, units='mks')

        # Truncate for better FFT
        required_length = int(1/delta_f * 1/dt)
        logging.warning('Rounding up time vectors')
        tt = tt[len(tt)-required_length:]
        hh = hh[len(hh)-required_length:]

        # For NRSur7dq4, NRSur7dq2, 2 Hz does not work (possibly, waveforms are too short)
        #tt, hh, dyn = sur(qq, chiA, chiB, dt=dt, f_low=2.0, f_ref=f_ref, ellMax=ellMax, M=M_tot, dist_mpc=luminosity_distance,
        #                  inclination=iota, phi_ref=phase, units='mks')



        # =================================================================== #
        # FFT, AFTER CONVERTING TO LAL TIMESERIES, AS OUT OF SimInspiralTD, CONDITIONING
        lal_epoch = geocent_time # Is it correct?????
        lal_f0 = np.min(frequencyvector)
        lal_hh_plus = lal.CreateREAL8TimeSeries('h_plus', lal_epoch, lal_f0, dt, 'strain', len(tt))
        lal_hh_cross = lal.CreateREAL8TimeSeries('h_cross', lal_epoch, lal_f0, dt, 'strain', len(tt))
        logging.warning('Taking -real(h) for h_+ and imag(h) for h_x. Should be opposite: (real(h), -imag(h))')
        lal_hh_plus.data.data = -np.real(hh)
        lal_hh_cross.data.data = np.imag(hh)

        # Data conditioning: option 1, that might change fRef
        # This is based on SimInspiralTDfromTD
        # Preparing variables
        f_min = frequencyvector[0,0] # Can be changed there, need to check
        extra_time_fraction = 0.1
        extra_cycles = 3.0
        textra = extra_cycles / f_min
        original_f_min = nrsur_f_low # f_min
        lal_mass_1 = mass_1 * lal.MSUN_SI * (1 + redshift)
        lal_mass_2 = mass_2 * lal.MSUN_SI * (1 + redshift)
        lal_S1z = chiA[-1] # also spin_1z
        lal_S2z = chiB[-1] # also spin_2z
        tchirp = lalsim.SimInspiralChirpTimeBound(f_min, lal_mass_1, lal_mass_2, lal_S1z, lal_S2z)
        fisco = 1.0 / (np.power(6.0, 1.5) * lal.PI * (lal_mass_1 + lal_mass_2) * lal.MTSUN_SI / lal.MSUN_SI);

        # Calling data conditioning
        lalsim.SimInspiralTDConditionStage1(lal_hh_plus, lal_hh_cross, extra_time_fraction * tchirp + textra, original_f_min)
        lalsim.SimInspiralTDConditionStage2(lal_hh_plus, lal_hh_cross, f_min, fisco)

        # [V - CHECK!] AT THIS STAGE WAVEFORMS AGREE IN CYCLES/AMPLITUDE
        #logging.warning('At this stage waveforms agree in cycles/amplitude with LAL')

        # COPYING _ht_postproccessing_SimInspiralTD()
        if True: # to keep padding from left, as in waveforms.py
            # This is done in LAL prior to a Fourier transform in SimInspiralFD, after SimInspiralTD is called.
            # Step 0 (even before SimInspiralTD). Set up a Nyqist frequency to be equal to maximum frequency
            # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L2895
            f_nyquist = f_max
            if int(np.log2(f_max/delta_f)) == np.log2(f_max/delta_f):
                #logging.warning('f_max/deltaF is not a power of two: changing f_max.')
                f_nyquist = 2**(np.floor(np.log2(f_nyquist/delta_f))-np.log2(1/delta_f))
                if f_max != f_nyquist:
                    raise ValueError('f_nyquist should be equal to f_max')

            # Step 1. Setting variable chirplen, and also frequency resolution (latter, if not set already)
            # Waveforms will be resized to chirplen. So, zeros added at the beginning (zero-padded) if
            # chirplen > h_plus.data.length, otherwise truncated
            # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L3023
            if delta_f==0.: # This would never be the case, if frequency array is set up
                # round length of time domain signal to next power of two
                pass
            else:
                chirplen = 2 * f_nyquist / delta_f
                if chirplen < lal_hh_plus.data.length: # This number should be h_plus.data.length
                    logging.warning('Specified frequency interval of %g Hz is too large for a chirp'+\
                                    'of duration %g s with Nyquist frequency %g Hz.'+\
                                    'The inspiral will be truncated.')

            # Step 3. Resizing the time-domain waveform to match some criteria above
            lal.ResizeREAL8TimeSeries(lal_hh_plus,
                                      int(lal_hh_plus.data.length - chirplen), int(chirplen))
            lal.ResizeREAL8TimeSeries(lal_hh_cross,
                                      int(lal_hh_cross.data.length - chirplen), int(chirplen))

        # FFT STAGE
        lal_hhf_plus = fft.fft_lal_timeseries(lal_hh_plus, delta_f)
        lal_hhf_cross = fft.fft_lal_timeseries(lal_hh_plus, delta_f)

        #new_frequencyvector = np.arange(0, lal_hhf_plus.deltaF*lal_hhf_plus.data.length, lal_hhf_plus.deltaF)
        #new_delta_f = lal_hhf_plus.deltaF
        print('HERE, check if delta_f is replaced: it should not be the case if we correctly truncated waveform above.')

        # At this stage, this is basically frequency-domain waveform out of LAL

        # To regular numpy array
        #hf_plus_out = lal_hhf_plus.data.data
        #hf_cross_out = lal_hhf_cross.data.data


        # ==== UTILS === : these are copied from LALTD_Waveform
        # FROM _update_frequency_range_indices()
        idx_low = int(f_min / delta_f) #int(f_min / delta_f)
        idx_high = int(f_max / delta_f)
        # FROM _lal_fd_strain_adjust_frequency_range()
        hf_cross_out = lal_hhf_cross.data.data[idx_low:idx_high+1]
        hf_plus_out = lal_hhf_plus.data.data[idx_low:idx_high+1]
        #new_frequencyvector = new_frequencyvector[idx_low:idx_high+1]

        # FROM _lal_fd_phase_correction_by_epoch_and_df()
        dt_correction = 1./delta_f #+ (lal_hhf_plus.epoch.gpsSeconds + lal_hhf_plus.epoch.gpsNanoSeconds * 1e-9)
        hf_plus_out *= np.exp(-1j * 2 * np.pi * dt_correction * np.squeeze(frequencyvector))
        hf_cross_out *= np.exp(-1j * 2 * np.pi * dt_correction * np.squeeze(frequencyvector))

        # _fd_phase_correction_geocent_time(): can be skipped??
        # I think we already corrected by geocent time...
        phi_in = np.exp(1.j*(2*np.squeeze(frequencyvector)*np.pi*geocent_time))
        hfp = phi_in * np.conjugate(hf_plus_out)  # it's already multiplied by the phase
        hfc = phi_in * np.conjugate(hf_cross_out)

        # FROM _fd_gwfish_output_format()
        hfp = hfp[:, np.newaxis] #hf_plus_out[:, np.newaxis]
        hfc = hfc[:, np.newaxis] # hf_cross_out[:, np.newaxis]
        return np.hstack((hfp, hfc)), frequencyvector # new_frequencyvector[:,np.newaxis]

        # =================================================================== #

        ## OLD!! FFT PROCEDURE (REPLACED BY ABOVE)

        ## Zero-padding data at the merger, not to erase a merger with a window
        #fft_roll_off = 0.2
        #zeropad_tt = np.arange(tt[-1] + dt, tt[-1] + dt + fft_roll_off, dt)
        #zeropad_hh = np.repeat(0, len(zeropad_tt))
        #tt = np.append(tt, zeropad_tt)
        #hh = np.append(hh, zeropad_hh)

        ## Fourier transforming time-domain strain
        ##hh_tilde, ff = fft(hh, dt, tt[0], tt[-1], roll_off=fft_roll_off)
        ## Minus sign following equations 2.5 and 2.6 from IMRPhenomXPHM paper,
        ## so that im matches exactly h_cross
        #hh_plus_tilde, ff = fft(np.real(hh), dt, tt[0], tt[-1], roll_off=fft_roll_off)
        #hh_cross_tilde, ff = fft(np.imag(hh), dt, tt[0], tt[-1], roll_off=fft_roll_off)
        ##print('Warning! Check hx convention')

        ## Matching user-defined frequencies
        ##tck_re = interpolate.splrep(ff, np.real(hh_tilde), s=0)
        ##tck_im = interpolate.splrep(ff, np.imag(hh_tilde), s=0)
        #tck_plus_re = interpolate.splrep(ff, np.real(hh_plus_tilde), s=0)
        #tck_plus_im = interpolate.splrep(ff, np.imag(hh_plus_tilde), s=0)
        #tck_cross_re = interpolate.splrep(ff, np.real(hh_cross_tilde), s=0)
        #tck_cross_im = interpolate.splrep(ff, np.imag(hh_cross_tilde), s=0)
        #hh_tilde_plus_re = interpolate.splev(frequencyvector, tck_plus_re, der=0)
        #hh_tilde_plus_im = interpolate.splev(frequencyvector, tck_plus_im, der=0)
        #hh_tilde_cross_re = interpolate.splev(frequencyvector, tck_cross_re, der=0)
        #hh_tilde_cross_im = interpolate.splev(frequencyvector, tck_cross_im, der=0)

        ##hh_tilde = hh_tilde_re + 1j * hh_tilde_im

        #hf_plus_out = hh_tilde_plus_re + 1j*hh_tilde_plus_im
        #hf_cross_out = hh_tilde_cross_re + 1j*hh_tilde_cross_im

        # =================================================================== #
        # OLD (REPLACED BY ABOVE)
        ## BORIS: weird Bilby correction: this is 100% needed to match complex phase with frequency-domain waveforms.
        ## Not doing so introduces extremely low mass errors!
        #delta_f = frequencyvector[1,0] - frequencyvector[0,0]
        #dt = 1. / delta_f #+ geocent_time
        #hf_plus_out *= np.exp(-1j * 2 * np.pi * dt * frequencyvector)
        #hf_cross_out *= np.exp(-1j * 2 * np.pi * dt * frequencyvector)

        ## Add initial 2pi*f*tc - phic - pi/4 to phase
        #phi_in = np.exp(1.j*(2*frequencyvector*np.pi*geocent_time))

        #hp = phi_in * np.conjugate(h_plus_out)  # it's already multiplied by the phase
        #hc = phi_in * np.conjugate(h_cross_out)

        #polarizations = np.hstack((hp, hc)) # original, gwsurrogate has already inverted hc because it gives hp-ihc
        #print('Warning: inverting hx in LAL caller')

        #return np.hstack((h_plus_out, h_cross_out))

    else:
        raise NotImplementedError('Frequency-domain surrogate waveforms are not implemented yet.')
        # The code below should work, but not tested
        tt, hh, dyn = sur(q, chiA, chiB, freqs=frequencyvector, f_ref=f_ref, ellMax=ellMax, M=M_tot, dist_mpc=luminosity_distance,
                          inclination=iota, phi_ref=phase, units='mks')
