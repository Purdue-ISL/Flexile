"""
CvarFlowAd_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the CvarFlowAd model in gurobi format.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging

def prepare_data(
    cap_file, tm_file, tm_index, tunnel_file, scenario_file):

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
      #print q,l,y[q,l]
  discon_prob = {}
  beta_factor_ = {}
  discon_pair = {}
  for i, j in sd_pairs:
    discon_prob[(i,j)] = 0
    for q in range(num_scenario):
      connected = 0
      for l in atunnels[i,j]:
        connected += y[q,l]
      if connected == 0:
        discon_prob[(i,j)] += s_probability[q]
        discon_pair[(i,j,q)] = 0
    beta_factor_[(i,j)] = 1 / (1 - beta - discon_prob[(i,j)])

  total_demand = 0
  total_pairs = 0
  for i, j in sd_pairs:
    total_pairs += 1
    total_demand += demand[i,j]

  # Gurobi Solver Model
  m = Model('teavar')

  # Variable definition
  x, s, t, alpha, ch = {}, {}, {}, {}, {}
  for l in range(anum_tunnel):
    for q in range(num_scenario):
      x[q,l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x[%s,%s]" % (q,l))

  for i, j in sd_pairs:
    alpha[i,j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha[%s,%s]" % (i,j))

  for q in range(num_scenario):
    for i, j in sd_pairs:
      s[i,j,q] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s[%s,%s,%s]" % (i,j,q))
      t[i,j,q] = m.addVar(lb=-100000, vtype=GRB.CONTINUOUS, name="t[%s,%s,%s]" % (i,j,q))

  cvar = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="cvar")
  max_alpha = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_alpha")

  beta_factor = 1 / (1-beta)
  # Objective definition
  #m.setObjective(cvar * 10000 + max_alpha, GRB.MINIMIZE)
  m.setObjective(cvar, GRB.MINIMIZE)

  # Constraints definition
  for i, j in sd_pairs:
    m.addConstr(
      cvar >= alpha[i,j] + beta_factor_[(i,j)] * (quicksum(s_probability[q] * s[i,j,q] for q in range(num_scenario))),
      "cvar_flow[%s,%s]" % (i,j) 
    )
    m.addConstr(
      max_alpha >= alpha[i,j],
      "max_alpha[%s,%s]" % (i,j) 
    )

  for q in range(num_scenario):
    for e1, e2 in arcs:
      m.addConstr(
        quicksum(x[q,l] for l in atraverse[e1, e2]) <= cap[e1, e2],
          "capacity[%s,%s,%s]" % (q, e1, e2)
      )

  for i, j in sd_pairs:
    for q in range(num_scenario):
      m.addConstr(
        s[i,j,q] >= t[i,j,q] - alpha[i,j],
        "loss_per_scenario[%s,%s,%s]" % (i,j,q) 
      )
      m.addConstr(
        s[i,j,q] >= 0,
        "positve_s[%s,%s,%s]" % (i,j,q) 
      )

  
  for i, j in sd_pairs:
    for q in range(num_scenario):
      if (i,j,q) in discon_pair:
        m.addConstr(
          t[i,j,q] == 0,
          "loss_per_flow_scenario[%s,%s,%s]" % (i,j,q) 
        )
      else:
        m.addConstr(
          t[i,j,q] == 1 - quicksum(x[q,l] * y[q,l] for l in atunnels[i,j])/demand[i,j],
          "loss_per_flow_scenario[%s,%s,%s]" % (i,j,q) 
        )

  m.update()
  return m

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
      y[l] = x[s,l].X
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

def compute_pct_loss(base_model, scenario_file, data):
  eps = 0.0000000001
  m = base_model.copy()

  sd_pairs = data['sd_pairs']
  num_scenario = data['num_scenario']
  anum_tunnel = data['anum_tunnel']
  s_probability = data['s_probability']
  # Variable retrieval
  x, s, t, alpha = {}, {}, {}, {}
  for l in range(anum_tunnel):
    for q in range(num_scenario):
      x[q,l] = m.getVarByName("x[%s,%s]" % (q,l))

  for i, j in sd_pairs:
    alpha[i,j] = m.getVarByName("alpha[%s,%s]" % (i,j))

  for q in range(num_scenario):
    for i, j in sd_pairs:
      s[i,j,q] = m.getVarByName("s[%s,%s,%s]" % (i,j,q))
      t[i,j,q] = m.getVarByName("t[%s,%s,%s]" % (i,j,q))

  # Solve
  #m.Params.method = 1
  m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  #m.Params.BarConvTol = 0.00001
  m.optimize()

  pa = post_analysis(x, data)

  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return pa, m.Runtime
  return None, m.Runtime
