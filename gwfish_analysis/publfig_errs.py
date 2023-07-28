import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer

import numpy as np
import pandas as pd

datadir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/'

labels = ['20230608_m','20230608_nom','20230608_mje']

parameters = pd.read_hdf('/fred/oz031/mem/gwmem_2022_container/image_content/pops_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj.hdf5')
fisher_parameters = "ra,dec,psi,theta_jn,luminosity_distance,mass_1,mass_2,a_1,a_2,geocent_time,phase,J_E,J_J".split(',')
fisher_parameters = [fp.replace('_','\_') for fp in fisher_parameters]

diagerrs = {}

for ll in labels:

  ff = datadir + ll + '_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_4000_diagerr.txt'

  diagerrs[ll] = np.loadtxt(ff)
  diagerrs[ll] = diagerrs[ll][:4000,:]

import ipdb; ipdb.set_trace()

plt.hist((diagerrs['20230608_m']/diagerrs['20230608_nom'])[:,0],bins=50,log=True,density=True, alpha=0.4)
plt.hist((diagerrs['20230608_m']/diagerrs['20230608_mje'])[:,0],bins=50,log=True,density=True, alpha=0.4)

cc = ChainConsumer()
cc.add_chain(diagerrs['20230608_m']/diagerrs['20230608_nom'], parameters=fisher_parameters)
#cc.add_chain(diagerrs['20230608_mje']/diagerrs['20230608_nom'], parameters=fisher_parameters)
cc.configure(kde=1.7)
fig = cc.plotter.plot_distributions()#log_scales=True)
plt.savefig('/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/test_err_dist.png')
plt.close()

import ipdb; ipdb.set_trace()
