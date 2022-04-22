#!/usr/bin/env python

"""

run.py

Run with python2.7+ (but not python3)

This file takes in a main configuration file and a topology configuration file, and outputs the computed reservation on each tunnel/sequence based on the scheme specified in the main configuration file.

Example usage: python run.py --main_config ../_config/main.yaml --topo_config ../_config/toy_config.yaml

"""
####
#### Imports
####
import sys
import time
import logging
import yaml
import argparse
from collections import defaultdict
import itertools
#import local gurobi path if needed.
sys.path.append('/package/gurobi/8.0.1/lib/python2.7/')
sys.path.append('..')
import module.models.solvers as solvers

####
#### Authorship information
####
__author__ = "Chuan Jiang"
__copyright__ = "Copyright 2021, Purdue ISL Flexile Project"
__credits__ = ["Chuan Jiang"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Chuan Jiang"
__email__ = "jiang486@purdue.edu"
__status__ = "beta"

def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--main_config", default="../_config/main.yaml",
                      help="main config file")
  parser.add_argument("--topo_config", default="../_config/toy_config.yaml",
                      help="topology config file")
  return parser.parse_args()

def _parse_configs(args):
  with open(args.main_config, 'r') as f:
    main_config = yaml.load(f)

  with open(args.topo_config, 'r') as f:
    topo_config = yaml.load(f)

  return {'main_config': main_config, 'topo_config': topo_config}

def _compute(main_config, topo_config):
  scheme = main_config['scheme']
  solver = None
  logger = logging.getLogger('main')
  #if 'output' not in main_config:
  #  main_config['output'] = 'output.txt'
  #  logger.warning('No output path provided. Use output.txt as default.')
  is_two_class = False
  if scheme == 'Teavar':
    solver = solvers.Teavar_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)  
  if scheme == 'CvarFlowSt':
    solver = solvers.CvarFlowSt_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)  
  if scheme == 'CvarFlowAd':
    solver = solvers.CvarFlowAd_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)  
  if scheme == 'Smore':
    solver = solvers.Smore_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None,
      is_smore_connected=False)
  if scheme == 'Smore_connected':
    solver = solvers.Smore_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None,
      is_smore_connected=True)
  if scheme == 'FlexileIP':
    solver = solvers.FlexileIP_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if scheme == 'FlexileBender':
    solver = solvers.FlexileBender_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if scheme == 'SwanThroughput':
    is_two_class = True
    solver = solvers.SwanThroughput_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if scheme == 'SwanMaxmin':
    is_two_class = True
    solver = solvers.SwanMaxmin_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if scheme == 'FlexileBender2Class':
    is_two_class = True
    solver = solvers.FlexileBender2Class_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if scheme == 'FlexileIP2Class':
    is_two_class = True
    solver = solvers.FlexileIP2Class_Solver(
      main_config=main_config,
      topo_config=topo_config,
      solver_config=None)
  if solver is None:   
    logger.error('WRONG scheme!')
    return
  if is_two_class:
    pct_loss_low, pct_loss_high, solving_time = solver.compute_pct_loss()
    logger.info('PctLoss low: %s %%, PctLoss high: %s %%, solving time: %s seconds' % (pct_loss_low * 100, pct_loss_high * 100, solving_time))
  else:
    pct_loss, solving_time = solver.compute_pct_loss()
    logger.info('PctLoss: %s %%, solving time: %s seconds' % (pct_loss * 100, solving_time))
  #logger.info('Tunnel reservation is saved in %s' % main_config['output'])

def _main(args, configs):
  main_config = configs['main_config']['main']
  topo_config = configs['topo_config']
  logging.basicConfig(level=main_config['log_level'])
  logger = logging.getLogger('main')

  logger.info("Start.")
  _compute(main_config, topo_config)
  logger.info("Done.")

if __name__ == "__main__":
  args = _parse_args()
  configs = _parse_configs(args)
  _main(args, configs)
