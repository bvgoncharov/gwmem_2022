"""
Comparison of the sxs waveform with NRSur7dq4 from gwsurrogate.
NRSur7dq4 is trained on the SXS catalog, so they should be the same.
Ultimately, to test adding memory corrections designed for sxs interpolated
waveforms to waveforms available in Bilby.
"""
import sxs
import bilby
import gwsurrogate

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

outdir = '../out_gwmem_2022/'

# Step 1. Loading the NRSur7dq4 waveform corresponding to GW150914, with gwsurrogate

#gwsurrogate.catalog.pull('NRSur7dq4')
sur = gwsurrogate.LoadSurrogate('NRSur7dq4')

q = 1.221
chiA = [3.415e-8, -4.091e-8, 0.3300] # from SXS catalog online, \Xi_{x,y,z}
chiB = [3.754e-8, 3.177e-8, -0.4399]
dt = 0.1        # timestep size, Units of M
f_low = 0       # initial frequency, f_low=0 returns the full surrogate
t_gws, h_gws, dyn_gws = sur(q, chiA, chiB, dt=dt, f_low=f_low)   # dyn stands for dynamics, do dyn.keys() to see contents

h_gws_sum = np.sum(np.array([val for val in h_gws.values()]),axis=0)

# Step 2. Loading the SXS waveform corresponding to GW150914

h_sxs = sxs.load("SXS:BBH:0305/Lev/rhOverM", extrapolation_order=3) # 0305 corresponds to GW150914
#h_with_memory = sxs.waveforms.memory.add_memory(h_sxs, integration_start_time=1000.0)

# Note, we can also access spherical harmonic modes. For example, the main (2,2) mode:
# h_sxs["Y_l2_m2.dat"][:,1] (+ polarization), h_sxs["Y_l2_m2.dat"][:,2] (x polarization)
# The zeroth column is a time vector

# There are 77 columns in h_sxs, I am not sure what all of these columns are.
# h_sxs_sum = np.sum(h_sxs, axis=1) # This might not be right.
h_sxs_sum_re = np.zeros(h_sxs.shape[0])
h_sxs_sum_im = np.zeros(h_sxs.shape[0])
for key in h_gws.keys():
  h_sxs_sum_re -= h_sxs["Y_l"+str(key[0])+"_m"+str(key[1])+".dat"][:,1]
  h_sxs_sum_im -= h_sxs["Y_l"+str(key[0])+"_m"+str(key[1])+".dat"][:,2]

# Selecting the peak time
t_max_sxs = h_sxs.t[h_sxs.index_closest_to(h_sxs.max_norm_time())]

# GWSURROGATE VS SXS
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
fig.suptitle('GW150914')
axs[0,0].set_title('Only (2,2) mode: IM')
axs[0,0].plot(h_sxs.t - t_max_sxs, h_sxs["Y_l2_m2.dat"][:,1], lw=0.3, label='SXS:0305')
axs[0,0].plot(t_gws, -np.imag(h_gws[(2,-2)]), ls='--', lw=0.3, label='NRSur7dq4, gwsurrogate')
axs[1,0].plot(h_sxs.t - t_max_sxs, h_sxs["Y_l2_m2.dat"][:,1], label='SXS:0305')
axs[1,0].plot(t_gws, -np.imag(h_gws[(2,-2)]), ls='--', label='NRSur7dq4, gwsurrogate')
axs[1,0].set_xlim([-100, 50])
axs[0,1].set_title('Only (2,2) mode: RE')
axs[0,1].plot(h_sxs.t - t_max_sxs, h_sxs["Y_l2_m2.dat"][:,2], lw=0.3, label='SXS:0305')
axs[0,1].plot(t_gws, -np.real(h_gws[(2,-2)]), ls='--', lw=0.3, label='NRSur7dq4, gwsurrogate')
axs[1,1].plot(h_sxs.t - t_max_sxs, h_sxs["Y_l2_m2.dat"][:,2], label='SXS:0305')
axs[1,1].plot(t_gws, -np.real(h_gws[(2,-2)]), ls='--', label='NRSur7dq4, gwsurrogate')
axs[1,1].set_xlim([-100, 50])
axs[0,2].set_title('All modes: IM')
axs[0,2].plot(h_sxs.t - t_max_sxs, h_sxs_sum_im, lw=0.3, label='SXS:0305')
axs[0,2].plot(t_gws, np.imag(h_gws_sum), ls='--', lw=0.3, label='NRSur7dq4, gwsurrogate')
axs[1,2].plot(h_sxs.t - t_max_sxs, h_sxs_sum_im, label='SXS:0305')
axs[1,2].plot(t_gws, np.imag(h_gws_sum), ls='--', label='NRSur7dq4, gwsurrogate')
axs[1,2].set_xlim([-100, 50])
axs[0,3].set_title('All modes: RE')
axs[0,3].plot(h_sxs.t - t_max_sxs, h_sxs_sum_re, lw=0.3, label='SXS:0305')
axs[0,3].plot(t_gws, np.real(h_gws_sum), ls='--', lw=0.3, label='NRSur7dq4, gwsurrogate')
axs[1,3].plot(h_sxs.t - t_max_sxs, h_sxs_sum_re, label='SXS:0305')
axs[1,3].plot(t_gws, np.real(h_gws_sum), ls='--', label='NRSur7dq4, gwsurrogate')
axs[1,3].set_xlim([-100, 50])
plt.legend()
plt.tight_layout()
plt.savefig(outdir + 'demo_comparison_1.png')
plt.close()

message = [
'IMPORTANT NOTES',
'To match SXS with gwsurrogate:',
'1. For (2,2) mode, I match column 1 in SXS as imag() and column 2 as real()',
'2. For (2,2) mode, I also invert both real() and imag() comparing to another waveform. So, SXS and gwsurrogate might be in counter-phase.',
'3. However, when I sum the modes, I match column 2 in SXS as imag(), and 1 as real(). I still keep the minus sign.',
'4. When I sum modes this way, I get imag() part still with some mismatch, and also significantly lower by magnitude than real().',
'It would be useful to understand these subtleties.'
]
for me in message: print(me)

# Step 3. Loading the NRSur7dq4 waveform corresponding to GW150914, with Bilby

duration = 2.
sampling_frequency = 4096
inj_par = dict(
    mass_ratio=1.221,
    chirp_mass=28.6,
    a_1=0.0,
    a_2=0.0,
    tilt_1=np.pi,
    tilt_2=np.pi,
    phi_jl=0.0,
    phi_12=0.0,
    luminosity_distance=440.0,
    theta_jn=np.pi,
    psi=0.0,
    phase=5.2220,
    geocent_time=1126259462.4,
    ra=0.82724876,
    dec=-1.3170442
)
waveform_arguments = dict(waveform_approximant='NRSur7dq4',
                          reference_frequency=50.)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=inj_par, waveform_arguments=waveform_arguments)
h_nrsur = waveform_generator.time_domain_strain()
t_nrsur = waveform_generator.time_array
