"""
This script can be ran outside of the environment
"""

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

n_events = 2000

#datadir = '/home/bgonchar/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'
datadir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'
scenarios = ['poincare','bms','ebms']
sc_labels = ['20230521_m','20230521_nom','20230521_mje']

detector_combinations = [['ET'],['ET','CE1'],['LLO','LHO','VIR'],['VOY','VIR']]

#outdir = '/home/bgonchar/out_gwmem_2022/publ_fig/'
outdir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/publ_fig/'

dict_z = {scl: {} for scl in sc_labels}
for scl in sc_labels:
  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)
    fname = datadir + scl + '_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(n_events-1)+'_' + dtcomb_label + '_200_noise_1_logzs.json'
    dict_z[scl][dtcomb_label] = pd.read_hdf(fname)

# ========= Next-generation detectors: spin memory ========== #
et_bms_bms_vs_ebms = dict_z['20230521_mje']['ET']['JE1JJ0'] - dict_z['20230521_mje']['ET']['JJ1JE1']
et_ebms_ebms_vs_bms = dict_z['20230521_m']['ET']['JE1JJ1'] - dict_z['20230521_m']['ET']['JJ0JE1']
etce_bms_bms_vs_ebms = dict_z['20230521_mje']['ETCE1']['JE1JJ0'] - dict_z['20230521_mje']['ETCE1']['JJ1JE1']
etce_ebms_ebms_vs_bms = dict_z['20230521_m']['ETCE1']['JE1JJ1'] - dict_z['20230521_m']['ETCE1']['JJ0JE1']

# Only for ET, some logBFs are < 0 when they are not supposed to be.
# Also, sometimes logBF = np.nan. This can be due to SVD issues.
# Proper fix (to-do): adjust SVD inversion threshold parameter, like for LISA.
# Quick fix below: remove injections with these values.
index_1 = et_bms_bms_vs_ebms[et_bms_bms_vs_ebms<0].index
index_2 = et_ebms_ebms_vs_bms[et_ebms_ebms_vs_bms<0].index
index_3 = et_ebms_ebms_vs_bms.isna()[et_ebms_ebms_vs_bms.isna()==True].index
index_4 = et_bms_bms_vs_ebms.isna()[et_bms_bms_vs_ebms.isna()==True].index
indices_bad = np.unique(np.concatenate([index_1,index_2,index_3,index_4]))
et_bms_bms_vs_ebms = et_bms_bms_vs_ebms.drop(index=indices_bad)
et_ebms_ebms_vs_bms = et_ebms_ebms_vs_bms.drop(index=indices_bad)
etce_bms_bms_vs_ebms = etce_bms_bms_vs_ebms.drop(index=indices_bad)
etce_ebms_ebms_vs_bms = etce_ebms_ebms_vs_bms.drop(index=indices_bad)
n_events = len(etce_ebms_ebms_vs_bms) # Overwriting

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.linspace(1,n_events,n_events,dtype=int)
ax.axhline(5., color='#f46036', lw=3, label='Detection')
ax.axhline(3., color='#f46036', linestyle='--', label='Strong evidence')
ax.loglog(indices, et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET',color='#2e294e',lw=0.8)
ax.loglog(indices, et_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#2e294e',ls=':')
ax.loglog(indices, etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET+CE',color='#1b998b',lw=0.8)
ax.loglog(indices, etce_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,color='#1b998b',ls=':')
prop = font_manager.FontProperties(**font)
ax.legend(prop=prop, handletextpad=0.02, columnspacing=0.1)
ax.set_xlim([1,n_events])
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$', fontdict=font)
ax.set_xlabel('Number of detections, sorted by $\ln\mathcal{B}$', fontdict=font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.savefig(outdir+'pop_cumulative_etce.pdf')
plt.close()

#geocent_time = parameters['geocent_time'].iloc[0:(largest_slurm_index+1)*opts.num]/60/60/24 # units days

# ============ Current-generation detectors: displacement memory ======== #

#lv_bms_bms_vs_p
#lv_ebms_ebms_vs_p
#lv_p_p_vs_bms
#lv_p_p_vs_ebms

import ipdb; ipdb.set_trace() 
