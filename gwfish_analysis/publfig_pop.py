"""
Previous steps:
- gwfish_memory_postprocessing.py
- combine_gwfish_memory_postprocessing.py

This script can be ran outside of the environment
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager

import numpy as np
import pandas as pd

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 20}

n_events = 1000

#datadir = '/home/bgonchar/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'
#datadir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'
datadir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/'
#datadir_lowrate = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230529_sorted_snrjj/'
datadir_lowrate = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/'
scenarios = ['poincare','bms','ebms']
#sc_labels = ['20230521_m','20230521_nom','20230521_mje']
#sc_labels = ['20230604_m','20230604_nom','20230604_mje']
sc_labels = ['20230608_m','20230608_nom','20230608_mje']
#sc_labels_lowrate = ['20230530_m','20230530_nom','20230530_mje']
sc_labels_lowrate = ['20230608_m','20230608_nom','20230608_mje']

#parameters = pd.read_hdf('/fred/oz031/mem/gwmem_2022_container/image_content/pops_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj.hdf5')
parameters = pd.read_hdf('/fred/oz031/mem/gwmem_2022_container/image_content/pops_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj.hdf5')
parameters_lowrate = pd.read_hdf('/fred/oz031/mem/gwmem_2022_container/image_content/pops_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj.hdf5')

detector_combinations = [['ET'],['ET','CE1'],['LLO','LHO','VIR'],['VOY','VIR']]

#outdir = '/home/bgonchar/out_gwmem_2022/publ_fig/'
outdir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/publ_fig/'

dict_z = {scl: {} for scl in sc_labels}
dict_z_lowrate = {scl: {} for scl in sc_labels_lowrate}
for scl, scllr in zip(sc_labels, sc_labels_lowrate):
  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)
    fname = datadir + scl + '_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(n_events-1)+'_' + dtcomb_label + '_200_noise_1_logzs.json'
    dict_z[scl][dtcomb_label] = pd.read_hdf(fname)
    # Low rate bound results
    fname_lowrate = datadir_lowrate + scllr + '_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(n_events-1)+'_' + dtcomb_label + '_200_noise_1_logzs.json'
    #if scllr != '20230530_mje':
    #  continue
    dict_z_lowrate[scllr][dtcomb_label] = pd.read_hdf(fname_lowrate)

# =========================================================== #
# ========= Next-generation detectors: spin memory ========== #
# =========================================================== #

et_bms_bms_vs_ebms = dict_z[sc_labels[2]]['ET']['JE1JJ0'] - dict_z[sc_labels[2]]['ET']['JJ1JE1']
et_ebms_ebms_vs_bms = dict_z[sc_labels[0]]['ET']['JE1JJ1'] - dict_z[sc_labels[0]]['ET']['JJ0JE1']
etce_bms_bms_vs_ebms = dict_z[sc_labels[2]]['ETCE1']['JE1JJ0'] - dict_z[sc_labels[2]]['ETCE1']['JJ1JE1']
etce_ebms_ebms_vs_bms = dict_z[sc_labels[0]]['ETCE1']['JE1JJ1'] - dict_z[sc_labels[0]]['ETCE1']['JJ0JE1']

# Only for ET, some logBFs are < 0 when they are not supposed to be.
# Also, sometimes logBF = np.nan. This can be due to SVD issues.
# Proper fix (to-do): adjust SVD inversion threshold parameter, like for LISA.
# Quick fix below: remove injections with these values.
# UPDATE: this has been fixed, commenting out this block.
#index_1 = et_bms_bms_vs_ebms[et_bms_bms_vs_ebms<0].index
#index_2 = et_ebms_ebms_vs_bms[et_ebms_ebms_vs_bms<0].index
#index_3 = et_ebms_ebms_vs_bms.isna()[et_ebms_ebms_vs_bms.isna()==True].index
#index_4 = et_bms_bms_vs_ebms.isna()[et_bms_bms_vs_ebms.isna()==True].index
#indices_bad = np.unique(np.concatenate([index_1,index_2,index_3,index_4]))
#et_bms_bms_vs_ebms = et_bms_bms_vs_ebms.drop(index=indices_bad)
#et_ebms_ebms_vs_bms = et_ebms_ebms_vs_bms.drop(index=indices_bad)
#etce_bms_bms_vs_ebms = etce_bms_bms_vs_ebms.drop(index=indices_bad)
#etce_ebms_ebms_vs_bms = etce_ebms_ebms_vs_bms.drop(index=indices_bad)
#n_events = len(etce_ebms_ebms_vs_bms) # Overwriting

# Low rate bound results
lr_et_bms_bms_vs_ebms = dict_z_lowrate[sc_labels_lowrate[2]]['ET']['JE1JJ0'] - dict_z_lowrate[sc_labels_lowrate[2]]['ET']['JJ1JE1']
lr_etce_bms_bms_vs_ebms = dict_z_lowrate[sc_labels_lowrate[2]]['ETCE1']['JE1JJ0'] - dict_z_lowrate[sc_labels_lowrate[2]]['ETCE1']['JJ1JE1']

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.loglog(indices, et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET',color='#2e294e',lw=0.8)
ax.loglog(indices, et_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#2e294e',ls=':')
ax.loglog(indices, etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
ax.loglog(indices, etce_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#1b998b',ls=':')
# Showing uncertainty between best-fit and lower merger rate
ax.fill_between(indices,y1=et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,y2=lr_et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,color='#2e294e',lw=0.0,alpha=0.5)
ax.fill_between(indices,y1=etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,y2=lr_etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,color='#1b998b',lw=0.0,alpha=0.5)
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([1,n_events])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Cumulative number of best detections', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.savefig(outdir+'pop_cumulative_etce.pdf')
plt.close()

# =================== Sorting my the observation time =================== #

geocent_time = parameters['geocent_time'].iloc[0:n_events]/60/60/24 # unit days
geocent_time = geocent_time.sort_values()
et_bms_bms_vs_ebms = et_bms_bms_vs_ebms.reindex(geocent_time.index)
et_ebms_ebms_vs_bms = et_ebms_ebms_vs_bms.reindex(geocent_time.index)
etce_bms_bms_vs_ebms = etce_bms_bms_vs_ebms.reindex(geocent_time.index)
etce_ebms_ebms_vs_bms = etce_ebms_ebms_vs_bms.reindex(geocent_time.index)
geocent_time = geocent_time.values
# For lower merger rate
geocent_time_lowrate = parameters_lowrate['geocent_time'].iloc[0:n_events]/60/60/24 # unit days
geocent_time_lowrate = geocent_time_lowrate.sort_values()
lr_et_bms_bms_vs_ebms = lr_et_bms_bms_vs_ebms.reindex(geocent_time_lowrate.index)
lr_etce_bms_bms_vs_ebms = lr_etce_bms_bms_vs_ebms.reindex(geocent_time_lowrate.index)
geocent_time_lowrate = geocent_time_lowrate.values

# Log plot
fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.loglog(geocent_time, et_bms_bms_vs_ebms.cumsum().values,label='ET',color='#2e294e',lw=0.8)
ax.loglog(geocent_time, et_ebms_ebms_vs_bms.cumsum().values,color='#2e294e',ls=':')
ax.loglog(geocent_time, etce_bms_bms_vs_ebms.cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
ax.loglog(geocent_time, etce_ebms_ebms_vs_bms.cumsum().values,color='#1b998b',ls=':')
# Showing uncertainty between best-fit and lower merger rate
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(et_bms_bms_vs_ebms.cumsum().values,lr_et_bms_bms_vs_ebms.cumsum().values[::-1]),color='#2e294e',lw=0.0,alpha=0.5)
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(etce_bms_bms_vs_ebms.cumsum().values,lr_etce_bms_bms_vs_ebms.cumsum().values[::-1]),color='#1b998b',lw=0.0,alpha=0.5)
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([np.min(geocent_time),np.max(geocent_time)])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Observation time [day]', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.savefig(outdir+'pop_cumulative_etce_time_log.pdf')
plt.close()

# Linear plot
fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.plot(geocent_time, et_bms_bms_vs_ebms.cumsum().values,label='ET',color='#2e294e',lw=0.8)
ax.plot(geocent_time, et_ebms_ebms_vs_bms.cumsum().values,color='#2e294e',ls=':')
ax.plot(geocent_time, etce_bms_bms_vs_ebms.cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
ax.plot(geocent_time, etce_ebms_ebms_vs_bms.cumsum().values,color='#1b998b',ls=':')
# Showing uncertainty between best-fit and lower merger rate
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(et_bms_bms_vs_ebms.cumsum().values,lr_et_bms_bms_vs_ebms.cumsum().values[::-1]),color='#2e294e',lw=0.0,alpha=0.5)
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(etce_bms_bms_vs_ebms.cumsum().values,lr_etce_bms_bms_vs_ebms.cumsum().values[::-1]),color='#1b998b',lw=0.0,alpha=0.5)
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([np.min(geocent_time),np.max(geocent_time)])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Observation time [day]', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
ax.set_xlim([0,200])
ax.set_ylim([0,20])
plt.savefig(outdir+'pop_cumulative_etce_time.pdf')
plt.close()

# ======================================================================= #
# ============ Current-generation detectors: displacement memory ======== #
# ======================================================================= #

#lv_bms_bms_vs_p
#lv_ebms_ebms_vs_p
#lv_p_p_vs_bms
#lv_p_p_vs_ebms

import ipdb; ipdb.set_trace() 
