from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

plt.rcParams['figure.constrained_layout.use'] = True

n_events = 2000

datadir = '/home/bgonchar/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'
scenarios = ['poincare','bms','ebms']
sc_labels = ['20230521_m','20230521_nom','20230521_mje']

detector_combinations = [['ET'],['ET','CE1'],['LLO','LHO','VIR'],['VOY','VIR']]

outdir = '/home/bgonchar/out_gwmem_2022/publ_fig/'

dict_z = {scl: {} for scl in sc_labels}
for scl in sc_labels:
  for dtcomb in detector_combinations:
    dtcomb_label = ''.join(dtcomb)
    fname = datadir + scl + '_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(n_events-1)+'_' + dtcomb_label + '_200_noise_1000_logzs.json'
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
ax.loglog(indices, et_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET, original BMS',color='#2e294e')
ax.loglog(indices, et_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,label='ET, extended BMS',color='#2e294e',ls=':')
ax.loglog(indices, etce_bms_bms_vs_ebms.sort_values(ascending=False).cumsum().values,label='ET+CE, original BMS',color='#1b998b')
ax.loglog(indices, etce_ebms_ebms_vs_bms.sort_values(ascending=False).cumsum().values,label='ET+CE, extended BMS',color='#1b998b',ls=':')
ax.axhline(5., color='#f46036', linestyle=':', label='Detection threshold')
ax.legend()
#plt.tight_layout()
ax.set_ylabel('Cumulative evidence, $\ln\mathcal{B}$')
ax.set_xlabel('Number of detections, sorted by $\ln\mathcal{B}$')
plt.savefig(outdir+'pop_cumulative_etce.png')
plt.close()

import ipdb; ipdb.set_trace() 
