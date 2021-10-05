"""
Teavar_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the Teavar model in gurobi format.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging

def prepare_data(
    cap_file, tm_file, tm_index, tunnel_file, scenario_file, all_scenario_file):

  # Index, Edge, Capacity definition
  nodes_set = set()
  arcs = []
  cap = {}
  logging.debug("cap_file: %s" % cap_file)
  with open(cap_file) as fin:
    for line in fin:
      if 'i' not in line:
        i, j, tcap = line.strip().split()
        tcap = float(tcap)
        nodes_set.add(i)
        nodes_set.add(j)
        if tcap > 0.0:
          arcs.append((i, j))
          cap[i,j] = tcap
  arcs_set = set(arcs)
  nodes = list(nodes_set)

  node_pairs = []
  atunnels = {}
  atraverse = {}
  demand = {}
  for i in nodes_set:
    for j in nodes_set:
      if i != j:
        node_pairs.append((i, j))
      atunnels[i,j] = set()
      atraverse[i,j] = set()
      demand[i,j] = 0

  # Demand definition
  sd_pairs = []
  logging.debug("tm_file: %s" % tm_file)
  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = float(tm)
        if int(h) == tm_index:
          demand[s,t] = tm
          if tm > 0.0:
            sd_pairs.append((s,t))
  sd_pairs_set = set(sd_pairs)

  # Tunnel node and edge set definition
  atunnel_edge_set = {}
  logging.debug("tunnel_file: %s" % tunnel_file)
  anum_tunnel = 0
  with open(tunnel_file) as fin:
    for line in fin:
      if 's' not in line:
        unpack = line.strip().split()
        s, t, k, edge_list = unpack
        edge_list = edge_list.split(',')
        atunnel_edge_set[anum_tunnel] = set()
        atunnels[s,t].add(anum_tunnel)
        for e in edge_list:
          u, v = e.split('-')
          atunnel_edge_set[anum_tunnel].add((u, v))
          atraverse[u,v].add(anum_tunnel)
        anum_tunnel = anum_tunnel + 1

  # Scenario definition
  failed_edge_set = {}
  s_probability = {}
  logging.debug("scenario_file: %s" % scenario_file)
  num_scenario = 0
  total_prob = 0
  with open(scenario_file) as fin:
    for line in fin:
      if 'edge' not in line:
        unpack = line.strip().split()
        edge_list, p = unpack
        if float(p) < 1e-6:
            continue
        total_prob += float(p)
        s_probability[num_scenario] = float(p)
        failed_edge_set[num_scenario] = set()
        if edge_list != 'no':
          edge_list = edge_list.split(',')
          for e in edge_list:
            u, v = e.split('-')
            failed_edge_set[num_scenario].add((u,v))
        num_scenario = num_scenario + 1
  s_probability[num_scenario] = 1-total_prob
  failed_edge_set[num_scenario] = set()
  for e1, e2 in arcs:
    failed_edge_set[num_scenario].add((e1, e2))
  num_scenario += 1

  # All Scenario calculation
  all_probability = 0
  logging.debug("all scenario_file: %s" % all_scenario_file)
  with open(all_scenario_file) as fin:
    for line in fin:
      if 'edge' not in line:
        unpack = line.strip().split()
        edge_list, p = unpack
        all_probability = all_probability + float(p)

  logging.debug("tcap: %s" % cap)
  logging.debug("demand: %s" % demand)
  logging.debug("arcs: %s" % arcs)
  logging.debug("node_pairs: %s" % node_pairs)
  logging.debug("atunnel_edge_set: %s" % atunnel_edge_set)
  logging.debug("atunnels: %s" % atunnels)
  logging.debug("atraverse: %s" % atraverse)
  logging.debug("anum_tunnel: %s" % anum_tunnel)
  logging.debug("failed_edge_set: %s" % failed_edge_set)
  logging.debug("num_scenario: %s" % num_scenario)
  logging.debug("s_probability: %s" % s_probability)

  return {"nodes": nodes, "nodes_set": nodes_set, "arcs": arcs,
          "arcs_set": arcs_set, "capacity": cap, "demand": demand,
          "sd_pairs": sd_pairs, "node_pairs": node_pairs,
          "atunnels": atunnels, "atraverse": atraverse, 
          "atunnel_edge_set": atunnel_edge_set, "anum_tunnel": anum_tunnel,
          "num_scenario": num_scenario, "failed_edge_set": failed_edge_set,
          "s_probability": s_probability, "all_probability": all_probability}

def create_base_model(beta, data):
  # Parameters
  nodes, arcs = data['nodes'], data['arcs']
  nodes_set, arcs_set = \
      data['nodes_set'], data['arcs_set']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  cap, demand = data['capacity'], data['demand']
  atunnels = data['atunnels']
  atraverse = data['atraverse']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  all_probability = data['all_probability']
  data['beta'] = beta

  y = {}
  rest_prob = 1
  for q in range(num_scenario):
    rest_prob = rest_prob - s_probability[q]
    for l in range(anum_tunnel):
      y[q,l] = 1
      for u,v in failed_edge_set[q]:
        if (u,v) in atunnel_edge_set[l]:
          y[q,l] = 0
          break

  # Gurobi Solver Model
  m = Model('Teavar')

  # Variable definition
  x, s, t = {}, {}, {}
  for l in range(anum_tunnel):
    x[l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x[%s]" % (l))

  for q in range(num_scenario):
    s[q] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s[%s]" % (q))
    for i, j in sd_pairs:
      t[i,j,q] = m.addVar(lb=-100000, vtype=GRB.CONTINUOUS, name="t[%s,%s,%s]" % (i,j,q))

  cvar = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="cvar")
  alpha = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")

  beta_factor = 1 / (1-beta)
  # Objective definition
  m.setObjective(cvar, GRB.MINIMIZE)

  # Constraints definition
  m.addConstr(
    cvar >= alpha + beta_factor*(quicksum(s_probability[q] * s[q] for q in range(num_scenario))),
        "cvar"
  )
  
  for e1, e2 in arcs:
    m.addConstr(
      quicksum(x[l] for l in atraverse[e1, e2]) <= cap[e1, e2],
        "capacity[%s,%s]" % (e1, e2)
    )

  for i, j in sd_pairs:
    for q in range(num_scenario):
      m.addConstr(
        s[q] >= t[i,j,q] - alpha,
        "loss_per_scenario[%s,%s,%s]" % (i,j,q) 
      )

  for q in range(num_scenario):
    m.addConstr(
      s[q] >= 0,
      "positve_s[%s]" % (q) 
    )
  
  for i, j in sd_pairs:
    for q in range(num_scenario):
      m.addConstr(
        t[i,j,q] == 1 - quicksum(x[l] * y[q,l] for l in atunnels[i,j])/demand[i,j],
        "loss_per_flow_scenario[%s,%s,%s]" % (i,j,q) 
      )

  m.update()
  return m

def post_analysis(x, all_scenario_file, data):
  eps = 0.000000001
  sd_pairs = data['sd_pairs']
  demand = data['demand']
  atunnels = data['atunnels']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  # Scenario definition
  failed_edge_set = {}
  y = {}
  loss = {}
  for i, j in sd_pairs:
    loss[i,j] = []
  with open(all_scenario_file) as fin:
    for line in fin:
      if 'edge' not in line:
        unpack = line.strip().split()
        edge_list, p = unpack
        s_probability = float(p)
        failed_edge_set = set()
        if edge_list != 'no':
          edge_list = edge_list.split(',')
          for e in edge_list:
            u, v = e.split('-')
            failed_edge_set.add((u,v))
        
        for l in range(anum_tunnel):
            y[l] = x[l].X
            for u,v in failed_edge_set:
                if (u,v) in atunnel_edge_set[l]:
                    y[l] = 0
                    break
        
        for i, j in sd_pairs:
            flow = 0
            for l in atunnels[i,j]:
                flow = flow + y[l]
            loss[i,j].append((1 - flow/demand[i,j], s_probability))
    logging.info("Post analysis:")
    scn = {}
    avail = {}
    for i, j in sd_pairs:
      loss[i,j].append((1.1, 1))
      loss[i,j].sort(key = lambda x:x[0])
      scn[i,j] = -1
      avail[i,j] = 0
    res = 1
    for loss_index in range(0, 100):
      cur_loss = loss_index * 0.01
      cur_avail = 1
      for i, j in sd_pairs:
        while loss[i,j][scn[i,j] + 1][0] <= cur_loss:
          scn[i,j] = scn[i,j] + 1
          avail[i,j] = avail[i,j] + loss[i,j][scn[i,j]][1]
        if avail[i,j] < cur_avail:
          cur_avail = avail[i,j]
      if cur_avail >= data['beta'] and res > cur_loss:
        res = cur_loss
      logging.info("Loss level: %s%%, Availability: %s%%" % (cur_loss * 100, cur_avail * 100))
    logging.info("Loss level: %s%%, Availability: %s%%" % (100, 100))
    return res

def compute_pct_loss(base_model, all_scenario_file, data):
  m = base_model.copy()

  sd_pairs = data['sd_pairs']
  num_scenario = data['num_scenario']
  anum_tunnel = data['anum_tunnel']
  # Variable retrieval
  alpha = m.getVarByName("alpha")
  x, s, t = {}, {}, {}
  for l in range(anum_tunnel):
    x[l] = m.getVarByName("x[%s]" % (l))

  for q in range(num_scenario):
    s[q] = m.getVarByName("s[%s]" % (q))
    for i, j in sd_pairs:
      t[i,j,q] = m.getVarByName("t[%s,%s,%s]" % (i,j,q))

  # Solve
  #m.Params.nodemethod = 2
  #m.Params.method = 1
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)

  print "alpha:"
  print alpha.X

  pa = post_analysis(x, all_scenario_file, data)

  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return pa, m.Runtime
  return None, None
