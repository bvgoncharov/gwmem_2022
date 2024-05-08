"""
Use this script after gwfish_memory_pipeline.py, with the same command line arguments,
an using the maximum --inj value, to check if all events that --inj ran over
are completed.
"""

import os
import tqdm

import gwfish_utils as gu

opts = gu.parse_commandline()
range_simulations = range(0, (opts.inj+1)*opts.num)

popdir, totaldir, namebase = gu.output_names(opts)

missing = []

for kk in tqdm.tqdm(range_simulations):
  outfile_name = namebase+'_'+str(kk)+'.pkl'
  if not os.path.exists(totaldir + outfile_name):
    #print('No result: ', kk)
    missing.append(kk)

if missing == []:
  print('No missing results')
else:
  print('Results missing from injections: ', missing)
