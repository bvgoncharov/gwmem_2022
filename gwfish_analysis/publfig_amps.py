import tqdm
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

labels = ['20230608_m', '20230606_nom', '20230608_mje']
true_jes = [1,0,1] # corresponding to above
true_jjs = [1,0,0] # m: both JE, JJ, mje: only JE, nom: none

for ll, true_je, true_jj in zip(labels, true_jes, true_jjs):
  fname = ll+'_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_999_ET_200_covjejj_noise.json'
  covjejj = pd.read_hdf(datadir+fname)
  n_events = len(covjejj)

  cov_inv = np.array([[covjejj['JE'],covjejj['JEJJ']],[covjejj['JEJJ'],covjejj['JJ']]])
  mean_param = np.array([true_je+covjejj['dJE'],true_jj+covjejj['dJJ']])[np.newaxis,:]

  pos_je = np.linspace(true_je-0.5,true_je+0.5,200)
  pos_jj = np.linspace(true_jj-0.5,true_jj+0.5,200)

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
  prob = prob/prob.max()

  log_prob_ii = log_prob_vals[0,:]
  prob_ii = np.exp(log_prob_ii - log_prob_ii.min())
  prob_ii = prob_ii/prob_ii.max()

  pos_je_grid = pos_je_grid.reshape(original_shape)
  pos_jj_grid = pos_jj_grid.reshape(original_shape)
  log_prob = log_prob.reshape(original_shape)
  prob = prob.reshape(original_shape)
  prob_ii = prob_ii.reshape(original_shape)

  # plt.contour(pos_je_grid,pos_jj_grid,prob)

  import ipdb; ipdb.set_trace()

  # Here, I need to meshgrid (A_s, A_d) values in (2xN) array
  # And then this needs to be passed to biv_normal_exponent, with transposition done
  # along the third axis.
  # This should give log-posterior for meshgrid values, N values, corresponding to events
  # Then I can just take a sum, and then log.


  # Then, make pyplot contours, or slices along respective A_d,s=1 or 0.
