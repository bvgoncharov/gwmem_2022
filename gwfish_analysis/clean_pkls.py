import os
import tqdm
import pickle

outdir = '/home/bgonchar/out_gwmem_2022/pop_max_o3_bbh_only_20230504_sorted_snrjj/'

all_files = os.listdir(outdir)
all_files = sorted([ff for ff in all_files if '.pkl' in ff])

for ff in tqdm.tqdm(all_files):
  with open(outdir + ff, 'rb') as fh:
    fm_object = pickle.load(fh)

  if '_lal_hlms' in dir(fm_object.derivative.waveform_object):
    fm_object.derivative.waveform_object._lal_hlms = None

  if '_lal_hf_plus' in dir(fm_object.derivative.waveform_object):
    fm_object.derivative.waveform_object._lal_hf_plus = None

  if '_lal_hf_cross' in dir(fm_object.derivative.waveform_object):
    fm_object.derivative.waveform_object._lal_hf_cross = None

  fm_object.pickle(outdir + ff)
