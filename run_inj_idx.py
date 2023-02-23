#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a full 15 parameter
space for an injected cbc signal. This is the standard injection analysis script
one can modify for the study of injected CBC events.

python /Users/boris.goncharov/projects/gwmem_2022/run_inj_idx.py --injf /fred/oz031/3g-pbh-pe/injection_tables/injection_table_ET1_ET2_ET3_SNR12_20210415.txt --waveform IMRPhenomXPHM --ifo_list ET1 ET2 ET3 --outdir /fred/oz031/out_3g-pbh-pe/ET123_phenxm_phm_0_20210415/ --num
"""
from __future__ import division, print_function
import os
kwan_path = '/home/kwan-yeung.ng/gitlab/bilby_3g'
if os.path.isdir(kwan_path):
  import sys
  sys.path.append(kwan_path)
  print('Added path: ', kwan_path)
import numpy as np
import bilby
import core_inj_idx as core

args = core.parse_command_line()

# Make sure output directory exists
if not os.path.exists(args.outdir):
	os.makedirs(args.outdir)

args = core.populate_args_from_injf(args)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = args.duration
sampling_frequency = args.fsample
sampler = 'dynesty'
# Specify the output directory and the name of the simulation.

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(args.seed)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
cosmo = bilby.gw.cosmology.DEFAULT_COSMOLOGY
m1src = args.m1src
m2src = args.m2src
mtotal = m1src+m2src
q=m2src/m1src
print(q)
tilt_1=0.
tilt_2=0.
a_1=0.
a_2=0.
phi_12=0.
phi_jl=0.
z=args.redshift
dL=cosmo.luminosity_distance(z).value
theta_jn_rad = args.theta_jn_deg*np.pi/180.0
ra_rad = args.ra_deg*np.pi/180.0
dec_rad = args.dec_deg*np.pi/180.0
psi_rad = args.psi_deg*np.pi/180.0
phase=0.0
gps_time=1577491218.0
m1_det = m1src*(1+z)
m2_det = m2src*(1+z)
Mc_det = bilby.gw.conversion.component_masses_to_chirp_mass(m1_det, m2_det)
total_mass_det = m1_det+m2_det
approx = args.waveform
label = core.bilby_output_label(mtotal, q, args.redshift, \
                                args.theta_jn_deg, approx)
bilby.core.utils.setup_logger(outdir=args.outdir, label=label)
fref=args.fref
fmin=args.fmin

# injection/simulation of a waveform
injection_parameters = dict(
	chirp_mass = Mc_det, mass_1=m1_det, mass_2=m2_det, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2,
	phi_12=phi_12, phi_jl=phi_jl, luminosity_distance=dL, theta_jn=theta_jn_rad, psi=psi_rad,
	phase=phase, geocent_time=gps_time, ra=ra_rad, dec=dec_rad, reference_frequency=fref, minimum_frequency=fmin)

# Fixed arguments passed into the source model
# approx: IMRPhenomPv2: standard, (2,2), IMRPhenomXPHM (l,m)=(2,2),...(4,4)
waveform_arguments = dict(waveform_approximant=approx, reference_frequency=fref, minimum_frequency=fmin)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
	duration=duration, sampling_frequency=sampling_frequency,
	frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
	parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
	waveform_arguments=waveform_arguments,start_time=injection_parameters['geocent_time'] - duration + 2.0)

# Set up interferometers.
ifos = bilby.gw.detector.InterferometerList(args.ifo_list)
for ifo in ifos:
	#ifo.set_strain_data_from_power_spectral_density(
	ifo.set_strain_data_from_zero_noise(
		sampling_frequency=sampling_frequency, duration=duration,
		start_time=injection_parameters['geocent_time'] - duration + 2.0)
	ifo.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)


# For this analysis, we implemenet the standard BBH priors defined, except for
# the definition of the time prior, which is defined as uniform about the
# injected value.
# Furthermore, we decide to sample in total mass and mass ratio, due to the
# preferred shape for the associated posterior distributions.
q_min=0.1
q_max=1.0
total_mass_min = total_mass_det*(0.25)
total_mass_max = total_mass_det*(2.5)
if approx=="IMRPhenomXPHM":
    if q>=0.8:
        chirp_mass_min = Mc_det*(0.75)
        chirp_mass_max = Mc_det*(1.25)
        total_mass_min = total_mass_det*(0.75)
        total_mass_max = total_mass_det*(1.25)
    if (q>=0.4)*(q<0.8):
        chirp_mass_min = Mc_det*(0.75)
        chirp_mass_max = Mc_det*(1.5)
        total_mass_min = total_mass_det*(0.75)
        total_mass_max = total_mass_det*(1.5)
    if q<0.4:
        chirp_mass_min = Mc_det*(0.5)
        chirp_mass_max = Mc_det*(2.0)
        total_mass_min = total_mass_det*(0.5)
        total_mass_max = total_mass_det*(2.0)
dist_min = dL/10.0
dist_max = dL*5.0
# We first output the prior ranges in a text file as a record tracker
BBHprior = \
"""mass_ratio = PowerLaw(name='mass_ratio', minimum=%f, maximum=%f, alpha=-2.0)
total_mass = Uniform(name='total_mass', minimum=%f, maximum=%f)
luminosity_distance = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=%f, maximum=%f)
dec = Cosine(name='dec')
ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
theta_jn = Sine(name='theta_jn')
psi = Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
delta_phase = Uniform(name='delta_phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
a_1 = Uniform(name='a_1', minimum=0, maximum=0.99)
a_2 = Uniform(name='a_2', minimum=0, maximum=0.99)
tilt_1 = Sine(name='tilt_1')
tilt_2 = Sine(name='tilt_2')
phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
geocent_time = Uniform(name='geocent_time', minimum=%f, maximum=%f)
""" %(q_min, q_max, total_mass_min, total_mass_max, dist_min, dist_max, gps_time-0.1, gps_time+0.1)

with open(args.outdir+'/%s.prior' %label,'w') as f_prior:
	f_prior.write(BBHprior)

priors = bilby.gw.prior.BBHPriorDict(args.outdir+'/%s.prior' %label)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
	interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
	distance_marginalization=True, phase_marginalization=args.margPhase, time_marginalization=True, distance_marginalization_lookup_table=args.outdir+'/%s_dist_lookup.npz' %label)
# Run sampler. In this case we're going to use the `dynesty` sampler
# The conversion function will determine the distance, phase and coalescence
# time posteriors in post processing.
if args.debug:
  #p0 = {pk: pv.sample() for pk, pv in likelihood.priors.items()}
  #likelihood.log_likelihood()
  import ipdb; ipdb.set_trace()
result = bilby.run_sampler(
	likelihood=likelihood, priors=priors, sampler=sampler, dlogz=0.1, nlive=2048, nact=10, maxmcmc=10000,
	injection_parameters=injection_parameters, outdir=args.outdir,
	label=label,
	conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

# Make a corner plot.
result.plot_corner()

