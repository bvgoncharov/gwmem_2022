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

lvk_p_p_vs_bms = dict_z[sc_labels[1]]['LLOLHOVIR']['JE0JJ0'] - dict_z[sc_labels[1]]['LLOLHOVIR']['JJ0JE1']
lvk_bms_bms_vs_p = dict_z[sc_labels[2]]['LLOLHOVIR']['JE1JJ0'] - dict_z[sc_labels[2]]['LLOLHOVIR']['JJ0JE0']
vv_p_p_vs_bms = dict_z[sc_labels[1]]['VOYVIR']['JE0JJ0'] - dict_z[sc_labels[1]]['VOYVIR']['JJ0JE1']
vv_bms_bms_vs_p = dict_z[sc_labels[2]]['VOYVIR']['JE1JJ0'] - dict_z[sc_labels[2]]['VOYVIR']['JJ0JE0']

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

lr_lvk_p_p_vs_bms = dict_z_lowrate[sc_labels[1]]['LLOLHOVIR']['JE0JJ0'] - dict_z_lowrate[sc_labels[1]]['LLOLHOVIR']['JJ0JE1']
lr_vv_p_p_vs_bms = dict_z_lowrate[sc_labels[1]]['VOYVIR']['JE0JJ0'] - dict_z_lowrate[sc_labels[1]]['VOYVIR']['JJ0JE1']

# Breaking the degeneracy between psi and phi, with SNR of odd-m SPH modes > 2
# 1/2. ET-CE
et_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_ET_1000_0.txt')
et_idx_snr_lss_2 = np.where(et_snr_sph_odd_m < 2)[0].tolist()
et_idx_snr_lss_2 = [ii for ii in et_idx_snr_lss_2 if ii<len(et_bms_bms_vs_ebms)]
et_bms_bms_vs_ebms.iloc[et_idx_snr_lss_2] = 0.
et_ebms_ebms_vs_bms.iloc[et_idx_snr_lss_2] = 0.
ce_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_CE1_1000_0.txt')
etce_snr_sph_odd_m = np.sqrt(et_snr_sph_odd_m**2 + ce_snr_sph_odd_m**2)
etce_idx_snr_lss_2 = np.where(etce_snr_sph_odd_m < 2)[0].tolist()
etce_bms_bms_vs_ebms.iloc[etce_idx_snr_lss_2] = 0.
etce_ebms_ebms_vs_bms.iloc[etce_idx_snr_lss_2] = 0.
lr_et_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_ET_1000_0.txt')
lr_et_idx_snr_lss_2 = np.where(lr_et_snr_sph_odd_m < 2)[0].tolist()
lr_et_idx_snr_lss_2 = [ii for ii in lr_et_idx_snr_lss_2 if ii<len(lr_et_bms_bms_vs_ebms)]
lr_et_bms_bms_vs_ebms.iloc[lr_et_idx_snr_lss_2] = 0.
lr_ce_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_CE1_1000_0.txt')
lr_etce_snr_sph_odd_m = np.sqrt(lr_et_snr_sph_odd_m**2 + lr_ce_snr_sph_odd_m**2)
lr_etce_idx_snr_lss_2 = np.where(lr_etce_snr_sph_odd_m < 2)[0].tolist()
lr_etce_idx_snr_lss_2 = [ii for ii in lr_etce_idx_snr_lss_2 if ii<len(lr_etce_bms_bms_vs_ebms)]
lr_etce_bms_bms_vs_ebms.iloc[lr_etce_idx_snr_lss_2] = 0.
# 2/2. LV-VOY
lho_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_LHO_1000_0.txt')
llo_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_LLO_1000_0.txt')
vir_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_VIR_1000_0.txt')
voy_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_VOY_1000_0.txt')
lv_snr_sph_odd_m = np.sqrt(lho_snr_sph_odd_m**2 + llo_snr_sph_odd_m**2 + vir_snr_sph_odd_m**2)
lv_idx_snr_lss_2 = np.where(lv_snr_sph_odd_m < 2)[0].tolist()
lv_idx_snr_lss_2 = [ii for ii in lv_idx_snr_lss_2 if ii<len(lvk_p_p_vs_bms)]
lvk_p_p_vs_bms.iloc[lv_idx_snr_lss_2] = 0.
lvk_bms_bms_vs_p.iloc[lv_idx_snr_lss_2] = 0.
vv_snr_sph_odd_m = np.sqrt(voy_snr_sph_odd_m**2 + vir_snr_sph_odd_m**2)
vv_idx_snr_lss_2 = np.where(vv_snr_sph_odd_m < 2)[0].tolist()
vv_idx_snr_lss_2 = [ii for ii in vv_idx_snr_lss_2 if ii<len(vv_p_p_vs_bms)]
vv_p_p_vs_bms.iloc[vv_idx_snr_lss_2] = 0.
vv_bms_bms_vs_p.iloc[vv_idx_snr_lss_2] = 0.
lr_lho_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_LHO_1000_0.txt')
lr_llo_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_LLO_1000_0.txt')
lr_vir_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_VIR_1000_0.txt')
lr_voy_snr_sph_odd_m = np.loadtxt('../../out_gwmem_2022/pop_min_o3_bbh_only_1yr_20230606_sorted_snrjj/snr_odd_sph_VOY_1000_0.txt')
lr_lv_snr_sph_odd_m = np.sqrt(lr_lho_snr_sph_odd_m**2 + lr_llo_snr_sph_odd_m**2 + lr_vir_snr_sph_odd_m**2)
lr_lv_idx_snr_lss_2 = np.where(lr_lv_snr_sph_odd_m < 2)[0].tolist()
lr_lv_idx_snr_lss_2 = [ii for ii in lr_lv_idx_snr_lss_2 if ii<len(lr_lvk_p_p_vs_bms)]
lr_lvk_p_p_vs_bms.iloc[lr_lv_idx_snr_lss_2] = 0.
lr_vv_snr_sph_odd_m = np.sqrt(lr_voy_snr_sph_odd_m**2 + lr_vir_snr_sph_odd_m**2)
lr_vv_idx_snr_lss_2 = np.where(lr_vv_snr_sph_odd_m < 2)[0].tolist()
lr_vv_idx_snr_lss_2 = [ii for ii in lr_vv_idx_snr_lss_2 if ii<len(lr_vv_p_p_vs_bms)]
lr_vv_p_p_vs_bms.iloc[lr_vv_idx_snr_lss_2] = 0.

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.loglog(indices, et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET',color='#2e294e',lw=0.8)
ax.loglog(indices, et_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#2e294e',ls=':')
ax.loglog(indices, etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
ax.loglog(indices, etce_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#1b998b',ls=':')
#ax.loglog(idx_snr_gtr_2, et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET',color='#2e294e',lw=0.8)
#ax.loglog(idx_snr_gtr_2, et_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#2e294e',ls=':')
#ax.loglog(idx_snr_gtr_2, etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
#ax.loglog(idx_snr_gtr_2, etce_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#1b998b',ls=':')
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

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.loglog(indices, lvk_p_p_vs_bms.sort_values(ascending=False).cumsum().values,label='LIGO+Virgo',color='#3D5656',lw=0.8)
ax.loglog(indices, lvk_bms_bms_vs_p.sort_values(ascending=False).cumsum().values,color='#3D5656',ls=':')
ax.loglog(indices, vv_p_p_vs_bms.sort_values(ascending=False).cumsum().values,label='Voyager+Virgo',color='#68B984',lw=0.8)
ax.loglog(indices, vv_p_p_vs_bms.sort_values(ascending=False).cumsum().values,color='#68B984',ls=':')
# Showing uncertainty between best-fit and lower merger rate
ax.fill_between(indices,y1=lvk_p_p_vs_bms.sort_values(ascending=False).cumsum().values,y2=lr_lvk_p_p_vs_bms.sort_values(ascending=False).cumsum().values,color='#3D5656',lw=0.0,alpha=0.5)
ax.fill_between(indices,y1=vv_p_p_vs_bms.sort_values(ascending=False).cumsum().values,y2=lr_vv_p_p_vs_bms.sort_values(ascending=False).cumsum().values,color='#68B984',lw=0.0,alpha=0.5)
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([1,n_events])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Cumulative number of best detections', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.savefig(outdir+'pop_cumulative_lvk.pdf')
plt.close()

# =================== Sorting my the observation time =================== #

geocent_time = parameters['geocent_time'].iloc[0:n_events]/60/60/24 # unit days
geocent_time = geocent_time.sort_values()
et_bms_bms_vs_ebms = et_bms_bms_vs_ebms.reindex(geocent_time.index)
et_ebms_ebms_vs_bms = et_ebms_ebms_vs_bms.reindex(geocent_time.index)
etce_bms_bms_vs_ebms = etce_bms_bms_vs_ebms.reindex(geocent_time.index)
etce_ebms_ebms_vs_bms = etce_ebms_ebms_vs_bms.reindex(geocent_time.index)
lvk_p_p_vs_bms = lvk_p_p_vs_bms.reindex(geocent_time.index)
lvk_bms_bms_vs_p = lvk_bms_bms_vs_p.reindex(geocent_time.index)
vv_p_p_vs_bms = vv_p_p_vs_bms.reindex(geocent_time.index)
vv_bms_bms_vs_p = vv_bms_bms_vs_p.reindex(geocent_time.index)
geocent_time = geocent_time.values
# For lower merger rate
geocent_time_lowrate = parameters_lowrate['geocent_time'].iloc[0:n_events]/60/60/24 # unit days
geocent_time_lowrate = geocent_time_lowrate.sort_values()
lr_et_bms_bms_vs_ebms = lr_et_bms_bms_vs_ebms.reindex(geocent_time_lowrate.index)
lr_etce_bms_bms_vs_ebms = lr_etce_bms_bms_vs_ebms.reindex(geocent_time_lowrate.index)
lr_lvk_p_p_vs_bms = lr_lvk_p_p_vs_bms.reindex(geocent_time_lowrate.index)
lr_vv_p_p_vs_bms = lr_vv_p_p_vs_bms.reindex(geocent_time_lowrate.index)
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
ax.set_ylim([0,15])
plt.savefig(outdir+'pop_cumulative_etce_time.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.plot(geocent_time, lvk_p_p_vs_bms.cumsum().values,label='LIGO+Virgo',color='#3D5656',lw=0.8)
ax.plot(geocent_time, lvk_bms_bms_vs_p.cumsum().values,color='#3D5656',ls=':')
ax.plot(geocent_time, vv_p_p_vs_bms.cumsum().values,label='Voyager+Virgo',color='#68B984',lw=0.8)
ax.plot(geocent_time, vv_bms_bms_vs_p.cumsum().values,color='#68B984',ls=':')
# Showing uncertainty between best-fit and lower merger rate
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(lvk_p_p_vs_bms.cumsum().values,lr_lvk_p_p_vs_bms.cumsum().values[::-1]),color='#3D5656',lw=0.0,alpha=0.5)
ax.fill(np.append(geocent_time,geocent_time_lowrate[::-1]),np.append(vv_p_p_vs_bms.cumsum().values,lr_vv_p_p_vs_bms.cumsum().values[::-1]),color='#68B984',lw=0.0,alpha=0.5)
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([np.min(geocent_time),np.max(geocent_time)])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Observation time [day]', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
#ax.set_xlim([0,300])
#ax.set_ylim([0,20])
plt.savefig(outdir+'pop_cumulative_lvk_time.pdf')
plt.close()

# ======================================================================= #
# ============ Current-generation detectors: displacement memory ======== #
# ======================================================================= #

#lv_bms_bms_vs_p
#lv_ebms_ebms_vs_p
#lv_p_p_vs_bms
#lv_p_p_vs_ebms

import ipdb; ipdb.set_trace() 
