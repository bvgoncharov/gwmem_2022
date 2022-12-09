import os
import gwsurrogate

nrsurfile = gwsurrogate.catalog.pull('NRSur7dq4')
nrsurfile2 = gwsurrogate.catalog.pull('NRHybSur3dq8')
nrsurpath = os.path.dirname(nrsurfile)+'/surrogate_downloads/'

print('Done! Now, run the following command and add it to your .bashrc:')
print('export LAL_DATA_PATH='+nrsurpath)
print('Make sure this path contains the .h5 file with the waveform model.')
