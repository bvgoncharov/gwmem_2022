import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def plot_waveform(waveform_obj, plot_name, fisco):
  """
  Makes a combined h(f) and h(t) plot.

  waveform_obj: GWFish waveform object.
  plot_name: output plot name with a full path.
  """

  # Matplotlib parameters
  co = 'blue'
  ls = '-'

  # Plot waveform
  f_start, f_end = waveform_obj.frequencyvector[0], waveform_obj.frequencyvector[-1]
  hf = waveform_obj.frequency_domain_strain
  fig, axs = plt.subplots(2,4, figsize=(20, 10), dpi=80)
  # Time-domain
  if hasattr(waveform_obj, 'lal_time_ht_plus'):
    htp = waveform_obj._lal_ht_plus.data.data
    htc = waveform_obj._lal_ht_cross.data.data
    tt = waveform_obj.lal_time_ht_plus
  elif hasattr(waveform_obj, 'time_domain_strain'):
    htp = waveform_obj.time_domain_strain[:,0]
    htc = waveform_obj.time_domain_strain[:,1]
    tt = waveform_obj.timevector
  else:
    htp, htc = None, None
  if htp is not None:
    axs[0,0].plot(tt, htp, label='h+', color=co, linestyle=ls)
    axs[0,0].set_xlim([-0.5,0.2])
    axs[1,0].plot(tt, htc, label='hx', color=co, linestyle=ls)
    axs[1,0].set_xlim([-0.15,0.15])
    axs[0,1].plot(tt, htp, label='h+', color=co, linestyle=ls)
    axs[0,1].set_xlim([-10,-1])
    axs[1,1].plot(tt, htc, label='hx', color=co, linestyle=ls)
    axs[1,1].set_xlim([tt[0],tt[0]+100])
  ## Frequency-domain
  axs[0,2].loglog(waveform_obj.frequencyvector, np.real(hf[:,0]), label='Re(h+)', color=co, linestyle=ls)
  #axs[0,2].set_xlim([f_start + 10,f_start + 20])
  axs[1,2].loglog(waveform_obj.frequencyvector, np.real(hf[:,1]), label='Re(hx)', color=co, linestyle=ls)
  axs[1,2].set_xlim([f_start + 10,f_start + 20])
  axs[0,3].semilogx(waveform_obj.frequencyvector, np.angle(hf[:,0] - 1j*hf[:,1]), label='Phase', alpha=0.3, color=co, linestyle=ls) # To be replace by np.angle()
  #axs[0,3].set_xlim([f_start + 10,f_start + 20])
  axs[1,3].loglog(waveform_obj.frequencyvector, np.abs(hf[:,0] - 1j*hf[:,1]), label='Abs', color=co, linestyle=ls) # To be replace by np.angle()
  axs[1,3].axvline(x=4*fisco, label='4fisco', linestyle=':',color='grey')
  axs[1,3].axvline(x=6.5*fisco, label='6.5fisco', linestyle='--',color='grey')
  axs[0,0].set_xlabel('Time [s]')
  axs[1,0].set_xlabel('Time [s]')
  axs[0,1].set_xlabel('Time [s]')
  axs[1,1].set_xlabel('Time [s]')
  axs[0,0].set_ylabel('hp')
  axs[1,0].set_ylabel('hx')
  axs[0,1].set_ylabel('hp')
  axs[1,1].set_ylabel('hx')
  axs[0,2].set_xlabel('Frequency [Hz]')
  axs[1,2].set_xlabel('Frequency [Hz]')
  axs[0,3].set_xlabel('Frequency [Hz]')
  axs[1,3].set_xlabel('Frequency [Hz]')
  axs[0,2].set_ylabel('hp')
  axs[1,2].set_ylabel('hx')
  axs[0,3].set_ylabel('Complex strain phase')
  axs[1,3].set_ylabel('Complex strain amplitude')
  plt.tight_layout()
  plt.legend()
  plt.savefig(plot_name)
  plt.close()
