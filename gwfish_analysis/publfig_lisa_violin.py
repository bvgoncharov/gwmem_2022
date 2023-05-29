import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#outdirs = [
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D1Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D1-5Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D2Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D2-5Gpc_20230515/',
#  #'/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D3Gpc_20230515/',
#  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_150_M1e6_D3-5Gpc_20230515/'
#]

outdirs = [
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1Gpc_20230518/',
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D1-5Gpc_20230518/',
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2Gpc_20230518/',
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D2-5Gpc_20230518/',
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3Gpc_20230518/',
  '/home/bgonchar/out_gwmem_2022/LISA_BBHs_1000_M1e6_D3-5Gpc_20230518/'
]

d_ls = [1.0, 1.5, 2.0, 2.5, 3.5] #3.0, 3.5] # Gpc

#labels = ['20230515_m','20230515_mje','20230515_nom']
labels = ['20230518_m','20230518_mje','20230518_nom']

#postfix = '_NRHybSur3dq8_gu.NRSurSPH_Memory_0.0001_0.0002_149_LISA_logzs.json'
postfix = '_NRHybSur3dq8_gu.NRSurSPH_Memory_0.0001_0.0002_999_LISA_logzs.json'

violin_kwargs = {
  'showextrema': False,
  'bw_method': 0.7,
}
facecolor = '#E39C12'
edgecolor = '#E39C12'
linewidth = 0.
alpha = 0.75

for lab in labels:
  ebms_vs_bms = []
  ebms_vs_poincare = []
  bms_vs_poincare = []
  spinmonly_vs_poincare = []
  for od, dl in zip(outdirs, d_ls):
    dict_z_file = od + lab + postfix
    with open(dict_z_file, 'r') as fj:
      list_logz_dicts = json.load(fj)['LISA']
  
    dict_z = pd.DataFrame.from_dict(list_logz_dicts)
    je1jj1_ebms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 1.0)' in kk][0]
    je0jj0_poincare_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 0.0)' in kk][0]
    je1jj0_bms_key = [kk for kk in dict_z.keys() if '(\'J_E\', 1.0)' in kk and '(\'J_J\', 0.0)' in kk][0]

    je0jj1_spinmonly_key = [kk for kk in dict_z.keys() if '(\'J_E\', 0.0)' in kk and '(\'J_J\', 1.0)' in kk][0]

    ebms_vs_bms.append( dict_z[je1jj1_ebms_key] - dict_z[je1jj0_bms_key] )
    ebms_vs_poincare.append( dict_z[je1jj1_ebms_key] - dict_z[je0jj0_poincare_key] )
    bms_vs_poincare.append( dict_z[je1jj0_bms_key] - dict_z[je0jj0_poincare_key] )

    spinmonly_vs_poincare.append( dict_z[je0jj1_spinmonly_key] - dict_z[je0jj0_poincare_key] )

  logbfs = [ebms_vs_bms, ebms_vs_poincare, bms_vs_poincare]
  names = ['ebms_vs_bms', 'ebms_vs_poincare', 'bms_vs_poincare']
  latex = ['$\ln\mathcal{B}^{\mathrm{eBMS}}_{BMS}$', \
           '$\ln\mathcal{B}^{\mathrm{eBMS}}_{Poincar{\'e}}$', \
           '$\ln\mathcal{B}^{\mathrm{eBMS}}_{Poincar{\'e}}$']
  for logbf, name, ltx in zip(logbfs, names, latex):
    fobj = plt.violinplot(np.array(logbf).T, positions=d_ls, **violin_kwargs)
    for pc in fobj['bodies']:
      pc.set_facecolor(facecolor)
      pc.set_edgecolor(edgecolor)
      pc.set_linewidth(linewidth)
      pc.set_alpha(alpha)
    #plt.legend()
    plt.xlabel('Luminosity distance [Gpc]')
    plt.ylabel('Log Bayes factor, $\ln\mathcal{B}$')
    plt.tight_layout()
    plt.savefig('/home/bgonchar/out_gwmem_2022/publ_fig/'+lab+'_'+name+'_lisa_logbf_dist.png')
    plt.close()

import ipdb; ipdb.set_trace()
