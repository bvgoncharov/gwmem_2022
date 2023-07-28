import tqdm
import numpy as np
import pandas as pd

from scipy import interpolate

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import plotting_utils as pu

def biv_normal_exponent(offset_from_mean,fisher_matrix):
  """
  Input: 
  - offset_from_mean: (1,2,N_events) array, where pairs are positions for
  where to evaluate the function (x,y).
  - fisher_matrix: (2,2,N_events), inverse of cov. matrix for events.
  """
  #temp = np.dot(mean_vector[:,np.newaxis],fisher_matrix)
  #temp = np.einsum('ijk,ikl->ik', np.transpose(aaa,axes=[2,0,1]), np.transpose(cov_inv,axes=[2,0,1]))
  temp = np.einsum('ijk,jlk->ilk',offset_from_mean,fisher_matrix)
  return -0.5*np.einsum('ijk,ljk->ilk',temp,offset_from_mean)[0,0,:]

def combined_posterior(covjejj, true_je, true_jj):

  n_events = len(covjejj)

  cov_inv = np.array([[covjejj['JE'],covjejj['JEJJ']],[covjejj['JEJJ'],covjejj['JJ']]])
  mean_param = np.array([true_je+covjejj['dJE'],true_jj+covjejj['dJJ']])[np.newaxis,:]

  #pos_je = np.linspace(true_je-0.1,true_je+0.1,200)
  #pos_jj = np.linspace(true_jj-0.75,true_jj+0.75,200)
  pos_je = np.linspace(true_je-0.25,true_je+0.25,500)
  pos_jj = np.linspace(true_jj-1.0,true_jj+1.0,500)

  pos_je_grid, pos_jj_grid = np.meshgrid(pos_je,pos_jj)
  original_shape = pos_je_grid.shape
  pos_je_grid = pos_je_grid.flatten()
  pos_jj_grid = pos_jj_grid.flatten()
  log_prob_vals = np.zeros((n_events,len(pos_je_grid)))

  for ii in tqdm.tqdm(range(len(pos_je_grid))):
    pairs_xy_list = np.repeat([[pos_je_grid[ii],pos_jj_grid[ii]]],n_events,axis=0).T[np.newaxis,:]
    log_prob_vals[:,ii] = biv_normal_exponent(pairs_xy_list-mean_param,cov_inv)

  log_prob = np.sum(log_prob_vals,axis=0)
  prob = np.exp(log_prob - log_prob.min())
  prob = prob/prob.sum()

  log_prob_ii = log_prob_vals[0,:]
  prob_ii = np.exp(log_prob_ii - log_prob_ii.min())
  prob_ii = prob_ii/prob_ii.sum()

  pos_je_grid = pos_je_grid.reshape(original_shape)
  pos_jj_grid = pos_jj_grid.reshape(original_shape)
  log_prob = log_prob.reshape(original_shape)
  prob = prob.reshape(original_shape)
  prob_ii = prob_ii.reshape(original_shape)

  return prob, pos_je_grid, pos_jj_grid, pos_je, pos_jj

def get_contours(zz, cred_levels=[0.997,0.954,0.682], n_steps=1000):
  """
  Given an array of z values, where z is some z(x,y), calculate
  z values corresponding to given credibility levels (default: 1,2,3 sigma).
  Here, z is a probability density function such that integral over it is unity.

  Based on: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution
  """
  tt = np.linspace(0, zz.max(), n_steps)
  integral = ((zz >= tt[:, None, None]) * zz).sum(axis=(1,2))
  ff = interpolate.interp1d(integral, tt)
  return ff(np.array(cred_levels))

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 20}

datadir = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj/'
parameters = pd.read_hdf('/fred/oz031/mem/gwmem_2022_container/image_content/pops_gwmem_2022/pop_max_o3_bbh_only_1yr_20230606_sorted_snrjj.hdf5')

labels = ['20230608_m', '20230608_nom', '20230608_mje']
true_jes = [1,0,1] # corresponding to above
true_jjs = [1,0,0] # m: both JE, JJ, mje: only JE, nom: none

#color_sequence = ['#FAF8F1','#FAEAB1','#E5BA73','#C58940']
color_sequence = ['#E4F5E5','#A6DFDE','#88A6E5','#8D6EC8']

for ll, true_je, true_jj in zip(labels, true_jes, true_jjs):
  print(ll)
  fname = ll+'_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_999_ET_200_covjejj_noise.json'
  fname_z1 = ll+'_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_9999_ET_200_49_covjejj_noise.json'
  covjejj = pd.read_hdf(datadir+fname)
  covjejj_z1 = pd.read_hdf(datadir+fname_z1)
  # Remove NaNs (usually last injections out of 10000, where snr could not be evaluated)
  covjejj_z1 = covjejj_z1.loc[~covjejj_z1.isnull().any(axis=1)]
  print('Number of NaNs: ', 10000-covjejj_z1.shape[0])

  prob, pos_je_grid, pos_jj_grid, pos_je, pos_jj = combined_posterior(covjejj, true_je, true_jj)
  prob_z1, pos_je_grid_z1, pos_jj_grid_z1, pos_je_z1, pos_jj_z1 = combined_posterior(covjejj_z1, true_je, true_jj)

  # Finding maximum aposteriori
  prob_je_max_idx, prob_jj_max_idx = np.where(prob==prob.max())
  prob_je_max_val = pos_je_grid[prob_je_max_idx[0],prob_jj_max_idx[0]]
  prob_jj_max_val = pos_jj_grid[prob_je_max_idx[0],prob_jj_max_idx[0]]

  fig = plt.figure(figsize=(8,6))
  axes = fig.add_subplot(111)
  img1 = plt.imshow(prob,origin='lower',extent=[pos_je.min(),pos_je.max(),pos_jj.min(),pos_jj.max()],cmap=pu.get_continuous_cmap(color_sequence),aspect='auto')
  c_all = plt.contour(pos_je_grid,pos_jj_grid,prob,get_contours(prob),colors='black',linewidths=0.5)
  c_all_z1 = plt.contour(pos_je_grid_z1,pos_jj_grid_z1,prob_z1,get_contours(prob_z1,cred_levels=[0.682]),colors='black',linewidths=1.5, linestyles=':')
  #c_0 = plt.contour(pos_je_grid,pos_jj_grid,prob_ii,get_contours(prob_ii),colors='#1b998b',linewidth=1,linestyles='--')
  axes.axvline(true_je,color='#f46036',linestyle='--')
  axes.axhline(true_jj,color='#f46036',linestyle='--')
  cb = plt.colorbar(img1,format='%.e')
  axes.set_xlabel('$A_\mathrm{d}$', fontdict=font)
  axes.set_ylabel('$A_\mathrm{s}$', fontdict=font)
  cb.set_label(label='Posterior probability density',size=font['size'],family=font['family'])
  cb.ax.tick_params(labelsize=font['size'])
  axes.tick_params(axis='y', labelsize = font['size'])
  axes.tick_params(axis='x', labelsize = font['size'])
  axes.set_xlim([prob_je_max_val-0.1,prob_je_max_val+0.1])
  axes.set_ylim([prob_jj_max_val-0.75,prob_jj_max_val+0.75])
  plt.savefig('/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/publ_fig/prob_jejj_'+ll+'.pdf')
  plt.close()

  # Here, I need to meshgrid (A_s, A_d) values in (2xN) array
  # And then this needs to be passed to biv_normal_exponent, with transposition done
  # along the third axis.
  # This should give log-posterior for meshgrid values, N values, corresponding to events
  # Then I can just take a sum, and then log.


  # Then, make pyplot contours, or slices along respective A_d,s=1 or 0.

  import ipdb; ipdb.set_trace()
