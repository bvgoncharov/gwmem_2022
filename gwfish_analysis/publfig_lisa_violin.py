import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 20}

#outdirs = [
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D1Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D1-5Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D2Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D2-5Gpc_20230515/',
#  #'/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D3Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D3-5Gpc_20230515/'
#]

#outdirs = [
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1Gpc_20230518/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1-5Gpc_20230518/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2Gpc_20230518/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2-5Gpc_20230518/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3Gpc_20230518/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3-5Gpc_20230518/'
#]

outdirs = [
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1-5Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2-5Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3Gpc_20230518/',
  '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3-5Gpc_20230518/'
]

d_ls = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] # Gpc

#labels = ['20230515_m','20230515_mje','20230515_nom']
labels = ['20230518_m','20230518_mje','20230518_nom']
label_order = [2,1,0]
plot_models_top_bottom = [['ebms_vs_poincare','ebms_vs_bms'],['bms_vs_poincare','ebms_vs_bms'],['bms_vs_poincare','ebms_vs_bms']]

#postfix = '_NRHybSur3dq8_gu.NRSurSPH_Memory_0.0001_0.0002_149_LISA_logzs.json'
postfix = '_NRHybSur3dq8_gu.NRSurSPH_Memory_0.0001_0.0002_999_LISA_logzs.json'

violin_kwargs = {
  'showextrema': False,
  'bw_method': 0.25,
}
facecolor = '#E39C12'
edgecolor = '#E39C12'
linewidth = 0.
alpha = 0.85

# Publication-quality plots
fig, axs = plt.subplots(2, 3, figsize=(12,6))
#ax0 = fig.add_subplot(111)
for lab, labidx, models in zip(labels, label_order, plot_models_top_bottom):
  log_bfs = {'ebms_vs_bms': [], 'ebms_vs_poincare': [], 'bms_vs_poincare': [], 'spinmonly_vs_poincare': []}
  #ebms_vs_bms = []
  #ebms_vs_poincare = []
  #bms_vs_poincare = []
  #spinmonly_vs_poincare = []
  for od, dl in zip(outdirs, d_ls):
    dict_z_file = od + lab + postfix
    with open(dict_z_file, 'r') as fj:
      list_logz_dicts = json.load(fj)['LISA']

    dict_z = pd.DataFrame.from_dict(list_logz_dicts)
    je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
    je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
    je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]

    je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]

    log_bfs['ebms_vs_bms'].append( dict_z[je1jj1_ebms_key].values - dict_z[je1jj0_bms_key].values )
    log_bfs['ebms_vs_poincare'].append( dict_z[je1jj1_ebms_key].values - dict_z[je0jj0_poincare_key].values )
    log_bfs['bms_vs_poincare'].append( dict_z[je1jj0_bms_key].values - dict_z[je0jj0_poincare_key].values )

    log_bfs['spinmonly_vs_poincare'].append( dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key] )
  #import ipdb; ipdb.set_trace()
  for ii, mm in enumerate(models): 

    # Replacing sign where logBFS are only negative
    if lab=='20230518_nom' and mm=='bms_vs_poincare' or lab=='20230518_mje' and mm=='ebms_vs_bms':
      y_vals = [-val for val in log_bfs[mm]]
    else:
      y_vals = log_bfs[mm]
    # Choosing different scales
    if not (lab=='20230518_nom' and mm=='ebms_vs_bms'):
      # Non-log Y scale only where both positive and negative logBFs
      y_vals = [np.log10(val) for val in y_vals]
      axs[ii,labidx].yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
      axs[ii,labidx].axhline(np.log10(5.), color='#f46036', lw=3)
      axs[ii,labidx].axhline(np.log10(3.), color='#f46036', linestyle='--')
    else:
      # Log Y scale where all logBFs are of the same sign
      # This is only eBMS vs BMS in Poincare scenario (model misspec.)
      axs[ii,labidx].set_ylim([-50,10])
      axs[ii,labidx].axhline(0, color='black',lw=0.5)

    fobj = axs[ii,labidx].violinplot(np.array(y_vals).T, positions=d_ls, **violin_kwargs)

    #fobj = axs[ii,labidx].violinplot(np.array(log_bfs[mm]).T, positions=d_ls, **violin_kwargs)

    axs[ii,labidx].tick_params(axis='y', labelsize = font['size'])
    axs[ii,labidx].tick_params(axis='x', labelsize = font['size'])
    #axs[ii,labidx].axhline(np.log10(5.), color='#f46036', lw=3)
    #axs[ii,labidx].axhline(np.log10(3.), color='#f46036', linestyle='--')
    for pc in fobj['bodies']:
      pc.set_facecolor(facecolor)
      pc.set_edgecolor(edgecolor)
      pc.set_linewidth(linewidth)
      pc.set_alpha(alpha)
#plt.legend()
#ax0.set_xlabel('Luminosity distance [Gpc]')
axs[0,0].set_title('Poincare Universe',fontdict=font)
axs[0,1].set_title('Original BMS Universe',fontdict=font)
axs[0,2].set_title('Extended BMS Universe',fontdict=font)
axs[1,1].set_xlabel('Luminosity distance [Gpc]', fontdict=font)
axs[0,0].set_ylabel('$\ln\mathcal{B}^{\mathrm{Poincar\\acute{e}}}_{\mathrm{BMS}}$', fontdict=font)
axs[1,0].set_ylabel('$\ln\mathcal{B}^{\mathrm{eBMS}}_{\mathrm{BMS}}$', fontdict=font)
axs[0,1].set_ylabel('$\ln\mathcal{B}^{\mathrm{BMS}}_{\mathrm{Poincar\\acute{e}}}$', fontdict=font)
axs[1,1].set_ylabel('$\ln\mathcal{B}^{\mathrm{BMS}}_{\mathrm{eBMS}}$', fontdict=font)
axs[0,2].set_ylabel('$\ln\mathcal{B}^{\mathrm{eBMS}}_{\mathrm{Poincar\\acute{e}}}$', fontdict=font)
axs[1,2].set_ylabel('$\ln\mathcal{B}^{\mathrm{eBMS}}_{\mathrm{BMS}}$', fontdict=font)
#plt.tight_layout()
plt.savefig('/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/publ_fig/lisa.pdf')
plt.close()

import ipdb; ipdb.set_trace()

# General overview plots
#for lab in labels:
#  ebms_vs_bms = []
#  ebms_vs_poincare = []
#  bms_vs_poincare = []
#  spinmonly_vs_poincare = []
#  for od, dl in zip(outdirs, d_ls):
#    dict_z_file = od + lab + postfix
#    with open(dict_z_file, 'r') as fj:
#      list_logz_dicts = json.load(fj)['LISA']
#  
#    dict_z = pd.DataFrame.from_dict(list_logz_dicts)
#    je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
#    je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
#    je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
#
#    je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
#
#    ebms_vs_bms.append( dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key] )
#    ebms_vs_poincare.append( dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key] )
#    bms_vs_poincare.append( dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key] )
#
#    spinmonly_vs_poincare.append( dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key] )
#
#  logbfs = [ebms_vs_bms, ebms_vs_poincare, bms_vs_poincare]
#  names = ['ebms_vs_bms', 'ebms_vs_poincare', 'bms_vs_poincare']
#  latex = ['$\ln\mathcal{B}^{\mathrm{eBMS}}_{BMS}$', \
#           '$\ln\mathcal{B}^{\mathrm{eBMS}}_{Poincar{\'e}}$', \
#           '$\ln\mathcal{B}^{\mathrm{eBMS}}_{Poincar{\'e}}$']
#  for logbf, name, ltx in zip(logbfs, names, latex):
#    fobj = plt.violinplot(np.array(logbf).T, positions=d_ls, **violin_kwargs)
#    for pc in fobj['bodies']:
#      pc.set_facecolor(facecolor)
#      pc.set_edgecolor(edgecolor)
#      pc.set_linewidth(linewidth)
#      pc.set_alpha(alpha)
#    #plt.legend()
#    plt.xlabel('Luminosity distance [Gpc]')
#    plt.ylabel('Log Bayes factor, $\ln\mathcal{B}$')
#    plt.tight_layout()
#    plt.savefig('/home/bgonchar/out_gwmem_2022/publ_fig/'+lab+'_'+name+'_lisa_logbf_dist.png')
#    plt.close()
#
#import ipdb; ipdb.set_trace()
