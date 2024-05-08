"""
This requires bilby
"""

import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}

outdir = '/hdfs/user/boris.goncharov/null_stream_out/'
#outdir = '/fred/oz031/null_stream_2021_out/'

duration = 2.
sampling_frequency = 4096.

H1 = bilby.gw.detector.get_empty_interferometer('H1')
H1.set_strain_data_from_zero_noise(sampling_frequency, duration) # no noise
L1 = bilby.gw.detector.get_empty_interferometer('L1')
L1.set_strain_data_from_zero_noise(sampling_frequency, duration)
V1 = bilby.gw.detector.get_empty_interferometer('V1')
V1.set_strain_data_from_zero_noise(sampling_frequency, duration)
LVC = bilby.gw.detector.networks.InterferometerList([H1, L1, V1])

inj_par = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1000., iota=0.4, psi=2.659,
    phase=1.3, geocent_time=1.9, ra=1.375, dec=1.2108, theta_jn=0.)

waveform_arguments = dict(waveform_approximant='IMRPhenomXPHM',
                          reference_frequency=50.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=inj_par, waveform_arguments=waveform_arguments)

# This sets spin-weighted spherical harmonic modes to include
waveform_generator.parameters['mode_array'] = [[2,2]]

waveform_generator.time_domain_strain()
