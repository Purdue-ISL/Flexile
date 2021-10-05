"""
Smore_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the SMORE model in gurobi format.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging

def prepare_data(cap_file, tm_file, tm_index, tunnel_file, scenario_file):

  # Index, Edge, Capacity definition
  nodes_set = set()
  arcs = []
  cap = {}
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
  logging.info("scenario_file: %s" % scenario_file)
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
  print s_probability[num_scenario], total_prob, s_probability[num_scenario] + total_prob
  failed_edge_set[num_scenario] = set()
  for e1, e2 in arcs:
    failed_edge_set[num_scenario].add((e1, e2))
  num_scenario += 1

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
          "s_probability": s_probability}

def post_analysis(x, data):
  eps = 0.000000001
  sd_pairs = data['sd_pairs']
  demand = data['demand']
  atunnels = data['atunnels']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  y = {}
  loss = {}
  for i, j in sd_pairs:
    loss[i,j] = []
  for s in range(num_scenario):
    for l in range(anum_tunnel):
      y[l] = x[s][l]
      for u,v in failed_edge_set[s]:
        if (u,v) in atunnel_edge_set[l]:
           y[l] = 0
           break
        
    for i, j in sd_pairs:
      flow = 0
      for l in atunnels[i,j]:
        flow = flow + y[l]
      loss[i,j].append((1 - flow/demand[i,j], s_probability[s]))
  eps = 0.0000001
  logging.info("Post analysis:")
  scn = {}
  avail = {}
  for i, j in sd_pairs:
    loss[i,j].append((1.1, 1, 100))
    loss[i,j].sort(key = lambda x:x[0])
    scn[i,j] = -1
    avail[i,j] = 0
  res = 1
  for loss_index in range(0, 100):
    cur_loss = loss_index * 0.01
    cur_avail = 1
    for i, j in sd_pairs:
      while cur_loss - loss[i,j][scn[i,j] + 1][0] > -eps:
        scn[i,j] = scn[i,j] + 1
        avail[i,j] = avail[i,j] + loss[i,j][scn[i,j]][1]
      if avail[i,j] < cur_avail:
        cur_avail = avail[i,j]
      if cur_avail >= data['beta'] and res > cur_loss:
        res = cur_loss
    logging.info("Loss level: %s%%, Availability: %s%%" % (cur_loss * 100, cur_avail * 100))
  logging.info("Loss level: %s%%, Availability: %s%%" % (100, 100))
  return res
  
def compute_pct_loss(data, beta, is_smore_connected):
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  anum_tunnel = data['anum_tunnel']
  data['beta'] = beta
  eps = 0.0000001
  x = {}
  labd_ = {}
  loss = {}
  prob = 0
  smore_res_list = []
  discon_list = []
  bad_prob = 0
  discon_prob = 0
  good_prob = 0
  throttle_prob = 0
  loss_list = []
  total_rt = 0
  for i in range(num_scenario):
    prob = prob + s_probability[i]
    x[i], mlu, rt = solve_once_mlu(data, failed_edge_set[i], is_smore_connected)
    total_rt += rt
  return post_analysis(x, data), rt

def solve_once_mlu(data, failed_edges, is_smore_connected):
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

  y = {}
  for l in range(anum_tunnel):
    y[l] = 1
    for u,v in failed_edges:
      if (u,v) in atunnel_edge_set[l]:
        y[l] = 0

  # Gurobi Solver Model
  m = Model('SMORE')

  # Variable definition
  x, ut = {}, {}
  for l in range(anum_tunnel):
    x[l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x[%s]" % (l))
  for u, v in arcs:
    ut[u,v] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="ut[%s,%s]" % (u,v))

  mlu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="mlu")

  # Objective definition
  m.setObjective(mlu, GRB.MINIMIZE)

  # Constraints definition
  for u, v in arcs:
    if (u,v) not in failed_edges:
      m.addConstr(
        mlu >= ut[u,v],
        "mlu[%s,%s]" % (u,v) 
      )

  for e1, e2 in arcs:
    alive = 1
    if (e1,e2) in failed_edges:
      alive = 0
    m.addConstr(
      quicksum(x[l] for l in atraverse[e1, e2]) == ut[e1,e2] * alive * cap[e1, e2],
        "capacity[%s,%s]" % (e1, e2)
    )
 
  disconnect = False 
  connected = 1
  for i, j in sd_pairs:
    for l in atunnels[i,j]:
      connected += y[l]
    if not is_smore_connected or connected > 0: 
      m.addConstr(
        quicksum(x[l] for l in atunnels[i,j]) == demand[i,j],
        "demand[%s,%s]" % (i,j) 
      )
    if connected > 0:
      pass
    else:
      disconnect = True

  # Solve
  #m.Params.nodemethod = 2
  m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()

  eps = 0.0000001

  logging.info("Runtime: %f seconds" % m.Runtime)
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    y = {}
    if mlu.X > 1:
      factor = mlu.X
    else:
      factor = 1
    for i in x:
      y[i] = x[i].X / factor
    for e1, e2 in arcs:
      if (e1, e2) in failed_edges:
        for l in atraverse[e1, e2]:
          y[l] = 0
    return y, m.ObjVal, m.Runtime
  else:
    for i in x:
      y[i] = 0
    return y, 1000000, m.Runtime
