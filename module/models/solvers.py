 #!/usr/bin/env python

"""

tunnel.py

Run with python2.7+ (but not python3)

This file contains the interfaces to call the solvers for models including Teavar.

"""

####
#### Imports
####
import sys
import logging

import module.models.Teavar_model as Teavar_model
import module.models.Smore_model as Smore_model
import module.models.FlexileIP_model as FlexileIP_model
import module.models.FlexileBender_model as FlexileBender_model

####
#### Authorship information
####
__author__ = "Chuan Jiang"
__copyright__ = "Copyright 2021, Purdue ISL PCF Project"
__credits__ = ["Chuan Jiang"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Chuan Jiang"
__email__ = "jiang486@purdue.edu"

class Teavar_Solver(object):
  def __init__(self, _sentinel=None, main_config=None, topo_config=None,
               solver_config=None):
    self.main_config = main_config
    self.topo_config = topo_config
    self.solver_config = solver_config
    self.logger = logging.getLogger("TeavarSolver")
    self.base_model = None
    self._parse_configs()

  def _parse_configs(self):
    assert self.topo_config is not None, "No topo config found!"
    self.cap_file = self.topo_config['data']['cap_file']
    self.tm_file = self.topo_config['data']['tm_file']
    self.tunnel_file = self.topo_config['data']['tunnel_file']
    self.scenario_file = self.topo_config['data']['scenario_file']
    self.beta = self.topo_config['attributes']['beta']
    self.tm_index = self.topo_config['traffic_matrix']['tm_index']

  def compute_pct_loss(self):
    if self.base_model == None:
      self.logger.debug("No base model. Creating base model")
      self.data = Teavar_model.prepare_data(
          self.cap_file, self.tm_file, self.tm_index,
          self.tunnel_file, self.scenario_file)
      self.base_model = Teavar_model.create_base_model(self.beta, self.data)
    return Teavar_model.compute_pct_loss(self.base_model, self.scenario_file, self.data)

class Smore_Solver(object):
  def __init__(self, _sentinel=None, main_config=None, topo_config=None,
               solver_config=None, is_smore_connected=False):
    self.main_config = main_config
    self.topo_config = topo_config
    self.solver_config = solver_config
    self.logger = logging.getLogger("SmoreSolver")
    self.base_model = None
    self.is_smore_connected = is_smore_connected
    self._parse_configs()

  def _parse_configs(self):
    assert self.topo_config is not None, "No topo config found!"
    self.cap_file = self.topo_config['data']['cap_file']
    self.tm_file = self.topo_config['data']['tm_file']
    self.tunnel_file = self.topo_config['data']['tunnel_file']
    self.scenario_file = self.topo_config['data']['scenario_file']
    self.beta = self.topo_config['attributes']['beta']
    self.tm_index = self.topo_config['traffic_matrix']['tm_index']

  def compute_pct_loss(self):
    if self.base_model == None:
      self.logger.debug("No base model. Creating base model")
      self.data = Smore_model.prepare_data(
          self.cap_file, self.tm_file, self.tm_index,
          self.tunnel_file, self.scenario_file)
    return Smore_model.compute_pct_loss(self.data, self.beta, self.is_smore_connected)

class FlexileIP_Solver(object):
  def __init__(self, _sentinel=None, main_config=None, topo_config=None,
               solver_config=None):
    self.main_config = main_config
    self.topo_config = topo_config
    self.solver_config = solver_config
    self.logger = logging.getLogger("FlexileIPSolver")
    self.base_model = None
    self._parse_configs()

  def _parse_configs(self):
    assert self.topo_config is not None, "No topo config found!"
    self.cap_file = self.topo_config['data']['cap_file']
    self.tm_file = self.topo_config['data']['tm_file']
    self.tunnel_file = self.topo_config['data']['tunnel_file']
    self.scenario_file = self.topo_config['data']['scenario_file']
    self.beta = self.topo_config['attributes']['beta']
    self.tm_index = self.topo_config['traffic_matrix']['tm_index']

  def compute_pct_loss(self):
    if self.base_model == None:
      self.logger.debug("No base model. Creating base model")
      self.data = FlexileIP_model.prepare_data(
          self.cap_file, self.tm_file, self.tm_index,
          self.tunnel_file, self.scenario_file)
    self.base_model = FlexileIP_model.create_base_model(self.beta, self.data)
    return FlexileIP_model.compute_pct_loss(self.base_model, self.scenario_file, self.data)

class FlexileBender_Solver(object):
  def __init__(self, _sentinel=None, main_config=None, topo_config=None,
               solver_config=None):
    self.main_config = main_config
    self.topo_config = topo_config
    self.solver_config = solver_config
    self.logger = logging.getLogger("FlexileBenderSolver")
    self.base_model = None
    self._parse_configs()

  def _parse_configs(self):
    assert self.topo_config is not None, "No topo config found!"
    self.cap_file = self.topo_config['data']['cap_file']
    self.tm_file = self.topo_config['data']['tm_file']
    self.tunnel_file = self.topo_config['data']['tunnel_file']
    self.scenario_file = self.topo_config['data']['scenario_file']
    self.beta = self.topo_config['attributes']['beta']
    self.step = self.topo_config['attributes']['step']
    self.tm_index = self.topo_config['traffic_matrix']['tm_index']

  def compute_pct_loss(self):
    if self.base_model == None:
      self.logger.debug("No base model. Creating base model")
      self.data = FlexileBender_model.prepare_data(
          self.cap_file, self.tm_file, self.tm_index,
          self.tunnel_file, self.scenario_file)
    return FlexileBender_model.benders_algorithm(self.beta, self.data, self.scenario_file, self.step)
