"""
Comparison of the sxs waveform with bilby waveform.
To make sure there is not a lot of difference due to systematic errors.
Ultimately, to test adding memory corrections designed for sxs interpolated
waveforms to bilby waveforms.
"""
import sxs
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

outdir = '/fred/oz031/mem/out_gwmem_2022/'

h = sxs.load("SXS:BBH:0305/Lev/rhOverM", extrapolation_order=3) # 0305 corresponds to GW150914
h_with_memory = sxs.waveforms.memory.add_memory(h, integration_start_time=1000.0)

plt.plot(h.t, h.norm)
plt.plot(h_with_memory.t, h_with_memory.norm)
plt.tight_layout()
plt.savefig(outdir + 'demo_comparison.png')
plt.close()
