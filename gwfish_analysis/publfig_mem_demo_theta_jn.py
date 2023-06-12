import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 18}

plt.rc('font', size=font['size'])

#fisco = 58.26002990860174

fig, axs = plt.subplots(2, 3, figsize=(12,6))
for ii, kk in zip(range(5), ['red','green','blue','black','purple']):

  td_strain_file = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/test_gw150914like_theta_20230606/20230606_m_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(ii)+'_td_waveform_mem.txt'
  fd_strain_file = '/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/test_gw150914like_theta_20230606/20230606_m_NRHybSur3dq8_gu.LALTD_SPH_Memory_9.0_20.0_'+str(ii)+'_fd_waveform_mem.txt'
  
  td_strain = np.loadtxt(td_strain_file)
  fd_strain = np.loadtxt(fd_strain_file)
  
  axs[0,0].plot(td_strain[:,0], np.real(td_strain[:,1] -1j*td_strain[:,2]), color=kk, lw=0.8) # total TD
  axs[0,1].plot(td_strain[:,0], np.abs(td_strain[:,3] -1j*td_strain[:,4]), color=kk, lw=0.8) # disp TD
  axs[0,2].plot(td_strain[:,0], np.abs(td_strain[:,5] -1j*td_strain[:,6]), color=kk, lw=0.8) # spin TD

  axs[1,0].loglog(fd_strain[:,0], fd_strain[:,1], color=kk, lw=0.8)
  axs[1,1].loglog(fd_strain[:,0], fd_strain[:,2], color=kk, lw=0.8)
  axs[1,2].loglog(fd_strain[:,0], fd_strain[:,3], color=kk, lw=0.8)

#disp_mem_approx = 7e-24*fd_strain[0,0]/fd_strain[:,0]
#axs[1,1].loglog(fd_strain[:,0],disp_mem_approx,color='yellow',ls=':')

axs[0,0].set_xlim([-1,0.1])
axs[0,1].set_xlim([-1,0.1])
axs[0,2].set_xlim([-1,0.1])
axs[0,0].set_ylabel('Strain, $h(t)$', fontdict=font)
axs[0,1].set_xlabel('Time [s]', fontdict=font)

axs[1,0].set_xlim([10,300])
axs[1,1].set_xlim([10,300])
axs[1,2].set_xlim([10,300])
axs[1,0].set_ylim([5e-25,5e-22])
axs[1,1].set_ylim([2e-25,2e-23])
axs[1,2].set_ylim([1e-27,1e-25])
axs[1,0].set_ylabel('Strain, $\\tilde{h}(f)$', fontdict=font)
axs[1,1].set_xlabel('Frequency [Hz]', fontdict=font)
axs[0,0].set_title('Total',fontdict=font)
axs[0,1].set_title('Disp. memory',fontdict=font)
axs[0,2].set_title('Spin memory',fontdict=font)
for ii in [0,1]:
  for jj in [0,1,2]:
    axs[ii,jj].tick_params(axis='y', labelsize = font['size'])
    axs[ii,jj].tick_params(axis='x', labelsize = font['size'])
plt.savefig('/fred/oz031/mem/gwmem_2022_container/image_content/out_gwmem_2022/publ_fig/mem_demo_theta_jn.pdf')
plt.close()

import ipdb; ipdb.set_trace()
