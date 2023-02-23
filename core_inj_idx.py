import argparse
import pandas as pd

import os
import bilby
import numpy as np

class ParFormats(object):
  """
  Translating command line parameter names to names used further on
  """
  def __init__(self):
    injpars = ['m1', 'm2', 'iota', 'ra', 'dec', 'psi', 'z', 'seglen', \
               'f_sample', 'f_low', 'f_ref']
    argspars = ['m1src','m2src','theta_jn_deg','ra_deg','dec_deg','psi_deg',\
                'redshift','duration','fsample','fmin','fref']
    self.pardict = dict(zip(injpars,argspars))

def parse_command_line():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num',type=int,default=0,help='Injection index')
  parser.add_argument('--injf',type=str,default='',help='Injection file')
  parser.add_argument('--m1src',type=float,default=None,help='Mass 1 at source')
  parser.add_argument('--m2src',type=float,default=None,help='Mass 2 at source')
  parser.add_argument('--theta_jn_deg',default=None,type=float,help='thetajn,deg')
  parser.add_argument('--ra_deg',default=None,type=float,help='ra,deg')
  parser.add_argument('--dec_deg',default=None,type=float,help='dec,deg')
  parser.add_argument('--psi_deg',default=None,type=float,help='psi,deg')
  parser.add_argument('--redshift',default=None,type=float,help='redshift')
  parser.add_argument('--duration',default=None,type=float,help='segment length')
  parser.add_argument('--fsample',default=None,type=float,\
                      help='sampling frequency (2*f_high)')
  parser.add_argument('--fmin',default=None,type=float,help='min frequency')
  parser.add_argument('--fref',default=None,type=float,help='ref frequency')
  parser.add_argument('--waveform',default=None,type=str,help='waveform model')
  parser.add_argument('--outdir', type=str, default='./',help='output folder')
  parser.add_argument('--margPhase', default=False, action='store_true',\
                      help='use this flag for phase-marginalization')
  parser.add_argument('--seed', type=int, default=1290643798,\
                      help='seed for RNG')
  parser.add_argument('--ifo_list', nargs="+", default='')
  parser.add_argument('--debug',type=int,default=0,\
                      help='Enter iPython debugger before sampling')

  return parser.parse_args()

def read_injf_with_pandas(args):
  return read_injf_with_pandas_general(args.injf)

def read_injf_with_pandas_general(file_name):
  """
  Reading injection table into Pandas data frame
  """
  with open(file_name,'r') as injf:
    header = injf.readline()
  injtab = pd.read_csv(file_name, names=header.replace('# ', '').split(), \
                       skiprows=1 ,delimiter=' ')
  return injtab.rename(mapper=ParFormats().pardict, axis='columns')

def populate_args_from_injf(args, verbose=True):
  injtab = read_injf_with_pandas(args)
  print('Populating arguments from an injection file, line ', args.num)
  for ag in args.__dict__.keys():
    if ag in injtab.keys() and args.__dict__[ag] is None:
      args.__dict__[ag] = injtab[ag][args.num]
      if verbose: print(ag, ': ', injtab[ag][args.num])
  if verbose:
    if args.margPhase:
      print('Marg Phase on')
    else:
      print('Marg Phase off')
  return args

def bilby_output_label(mtotal, q, redshift, theta_jn_deg, approx, 
                       radec=None, label_suffix=''):
  if radec:
    return 'mtotal%.1f_q%.1f_z%.1f_iota%.1f_ra%.1f_dec%.1f_%s' \
           %(mtotal, 1./q, redshift, theta_jn_deg, radec[0], \
           radec[1], approx) + label_suffix
  else:
    return 'mtotal%.1f_q%.1f_z%.1f_iota%.1f_%s' %(mtotal, 1./q, redshift, \
            theta_jn_deg, approx) + label_suffix

def bilby_output_label_from_args(args, radec = None, label_suffix = ''):
  return bilby_output_label(args.m1src+args.m2src, args.m2src/args.m1src,\
                            args.redshift, args.theta_jn_deg, \
                            args.waveform, radec = radec, \
                            label_suffix = label_suffix)

class Result(object):
  """ Result object """
  def __init__(self, args, plot_corner = False, load_result = False,
               load_checkpoint_if_no_result = False, radec = None, 
               injkeys = ['m1src','m2src','redshift','theta_jn_deg'],
               pars = ['redshift', 'luminosity_distance'], 
               label_suffix = ''):
    self.args = args
    self.plot_corner = plot_corner
    self.load_result = load_result
    self.load_checkpoint = load_checkpoint_if_no_result
    self.radec = radec
    self.injkeys = injkeys
    self.pars = pars
    self.label_suffix = label_suffix

    # Internal variables
    self.result_objs = []
    self.rec_val = dict.fromkeys(pars, [])
    self.rec_err = dict.fromkeys(pars, [])
    self.need_to_continue = []
    self.need_to_start = []

    self.injtab = read_injf_with_pandas(args)
    # Use --num -1 to loop through all results
    if args.num != -1:
      self.injf_lines = [args.num]
    else:
      self.injf_lines = [ii for ii in range(len(self.injtab.index))]

  def main_pipeline(self, archive_pars = []):
    for ll in self.injf_lines:
      self.args.num = ll
      self.args = populate_args_from_injf(self.args, verbose=False)
      self.populate_inj_vals()
      self.label = bilby_output_label_from_args(self.args, \
                                radec = self.radec, \
                                label_suffix = self.label_suffix)
      self.basename = self.args.outdir + '/' + self.label
      success = self.load_result_if_exists()
      if success:
        self.to_do_if_result_exists()
        if self.plot_corner:
          self.result_objs[-1].plot_corner()
        if archive_pars is not []:
          self.archive_posteriors(archive_pars)

  def load_result_if_exists(self):
    if os.path.isfile(self.basename + '_result.json'):
      print(self.args.num,' completed: ', self.basename + '_result.json')
      if self.load_result:
        self.result_objs.append( bilby.result.read_in_result(\
                                 filename=self.basename + '_result.json') )
        return True
      else:
        return False
    elif os.path.isfile(self.basename + '_resume.pickle'):
      print(self.args.num, ' is incomplete')
      self.need_to_continue.append(self.args.num)
      if self.load_checkpoint:
        self.result_objs.append( bilby.result.read_in_result(\
                                 filename=self.basename + '_resume.pickle') )
        return True
      else:
        return False
    else:
      print(self.args.num, ' has not produced any checkpoints, ')
      self.need_to_start.append(self.args.num)
      return False

  def to_do_if_result_exists(self):
    for kk in self.pars:
      self.rec_val[kk].append( np.median(self.result_objs[-1].posterior[kk]) )
      self.rec_err[kk].append( \
           np.percentile(self.result_objs[-1].posterior[kk], 84, axis=0) - \
           np.percentile(self.result_objs[-1].posterior[kk], 16, axis=0) )

  def populate_inj_vals(self):
    """ Populate injection values from command line args """
    self.iv = dict.fromkeys(self.injkeys, [])
    for kk in self.iv.keys():
      if kk in self.args.__dict__.keys():
        self.iv[kk].append( self.args.__dict__[kk] )
      else:
        error_message = kk + ' is not in command line args'
        raise ValueError(error_message)

  def print_injtab(self):
    print(self.injtab.to_string())

  def archive_posteriors(self, pars_save):
    with open(self.args.outdir+'/archive_'+self.label+'.txt', 'w') as outf:
      self.result_objs[-1].posterior[pars_save].to_csv(outf, index=False)
