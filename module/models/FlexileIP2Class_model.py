"""
FlexileIP2Class_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the FlexileIP2Class model in gurobi format.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging

def prepare_data(
    cap_file, tm_file, tm_index_low, tm_index_high, tunnel_file, scenario_file, scale_low):

  # Index, Edge, Capacity definition
  min_tf = 0 
  nodes_set = set()
  arcs = []
  cap = {}
  weight = {}
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
  btunnels = {}
  btraverse = {}
  demand_low = {}
  demand_high = {}
  for i in nodes_set:
    for j in nodes_set:
      if i != j:
        node_pairs.append((i, j))
      atunnels[i,j] = set()
      btunnels[i,j] = set()
      atraverse[i,j] = set()
      btraverse[i,j] = set()
      demand_low[i,j] = 0
      demand_high[i,j] = 0

  # Demand definition
  sd_pairs = []
  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = max(float(tm) * scale_low, min_tf)
        if int(h) == tm_index_low:
          demand_low[s,t] = tm
          if tm > 0.0:
            sd_pairs.append((s,t))

  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = max(float(tm), min_tf)
        if int(h) == tm_index_high:
          demand_high[s,t] = tm
          if tm > 0.0 and (s,t) not in sd_pairs:
            sd_pairs.append((s,t))
  sd_pairs_set = set(sd_pairs)

  # Tunnel node and edge set definition
  atunnel_edge_set = {}
  atunnel_length = {}
  anum_tunnel = 0
  with open(tunnel_file) as fin:
    for line in fin:
      if 's' not in line:
        unpack = line.strip().split()
        s, t, k, edge_list, leng = unpack
        # whether to consider for lantency-sensitive
        lt_sensitive = int(k) < 3
        edge_list = edge_list.split(',')
        atunnel_edge_set[anum_tunnel] = set()
        atunnels[s,t].add(anum_tunnel)
        if lt_sensitive:
            btunnels[s,t].add(anum_tunnel)
        leng = 0
        for e in edge_list:
          u, v = e.split('-')
          atunnel_edge_set[anum_tunnel].add((u, v))
          atraverse[u,v].add(anum_tunnel)
          if lt_sensitive:
              btraverse[u,v].add(anum_tunnel)
        atunnel_length[anum_tunnel] = leng
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
        else:
          no_failure_case = num_scenario
        num_scenario = num_scenario + 1
  s_probability[num_scenario] = 1-total_prob
  failed_edge_set[num_scenario] = set()
  for e1, e2 in arcs:
    failed_edge_set[num_scenario].add((e1, e2))
  num_scenario += 1

  # perfect Scenario definition
  perfect_scenario_set = {}
  num_perfect = 0
  prob_perfect = 0

  # All Scenario calculation
  all_probability = 0
  logging.info("all_scenario_file: %s" % scenario_file)
  with open(scenario_file) as fin:
    for line in fin:
      if 'edge' not in line:
        unpack = line.strip().split()
        edge_list, p = unpack
        all_probability = all_probability + float(p)

  logging.debug("tcap: %s" % cap)
  logging.debug("demand_low: %s" % demand_low)
  logging.debug("arcs: %s" % arcs)
  logging.debug("node_pairs: %s" % node_pairs)
  logging.debug("atunnel_edge_set: %s" % atunnel_edge_set)
  logging.debug("atunnels: %s" % atunnels)
  logging.debug("btunnels: %s" % btunnels)
  logging.debug("atraverse: %s" % atraverse)
  logging.debug("anum_tunnel: %s" % anum_tunnel)
  logging.debug("failed_edge_set: %s" % failed_edge_set)
  logging.debug("num_scenario: %s" % num_scenario)
  logging.debug("s_probability: %s" % s_probability)

  return {"nodes": nodes, "nodes_set": nodes_set, "arcs": arcs,
          "arcs_set": arcs_set, "capacity": cap, 
          "demand_low": demand_low, "demand_high": demand_high,
          "sd_pairs": sd_pairs, "node_pairs": node_pairs,
          "atunnels": atunnels, "atraverse": atraverse, 
          "btunnels": btunnels, "btraverse": btraverse, 
          "atunnel_edge_set": atunnel_edge_set, "anum_tunnel": anum_tunnel,
          "atunnel_length": atunnel_length, 
          "num_scenario": num_scenario, "failed_edge_set": failed_edge_set,
          "s_probability": s_probability, "all_probability": all_probability,
          "perfect_scenario_set": perfect_scenario_set, 
          "prob_perfect": prob_perfect,
          "no_failure_case": no_failure_case}

def create_base_model(beta_low, beta_high, data):
  # Parameters
  nodes, arcs = data['nodes'], data['arcs']
  nodes_set, arcs_set = \
      data['nodes_set'], data['arcs_set']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  cap, demand_low, demand_high = data['capacity'], data['demand_low'], data['demand_high']
  atunnels = data['atunnels']
  atraverse = data['atraverse']
  btunnels = data['btunnels']
  btraverse = data['btraverse']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  all_probability = data['all_probability']
  perfect_scenario_set = data["perfect_scenario_set"]
  prob_perfect = data["prob_perfect"]
  data["beta_low"] = beta_low
  data["beta_high"] = beta_high

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
  m = Model('teavar')

  # Variable definition
  x1, s1, t1, z1, x2, s2, t2, z2 = {}, {}, {}, {}, {}, {}, {}, {}
  for l in range(anum_tunnel):
    for q in range(num_scenario):
      x1[q,l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1[%s,%s]" % (q,l))
      x2[q,l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x2[%s,%s]" % (q,l))

  for q in range(num_scenario):
    for i, j in sd_pairs:
      z1[i,j,q] = m.addVar(lb=0.0, vtype=GRB.BINARY, name="z1[%s,%s,%s]" % (i,j,q))
      s1[i,j,q] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s1[%s,%s,%s]" % (i,j,q))
      t1[i,j,q] = m.addVar(lb=-100000, vtype=GRB.CONTINUOUS, name="t1[%s,%s,%s]" % (i,j,q))
      z2[i,j,q] = m.addVar(lb=0.0, vtype=GRB.BINARY, name="z2[%s,%s,%s]" % (i,j,q))
      s2[i,j,q] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s2[%s,%s,%s]" % (i,j,q))
      t2[i,j,q] = m.addVar(lb=-100000, vtype=GRB.CONTINUOUS, name="t2[%s,%s,%s]" % (i,j,q))

  alpha1 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha1")
  alpha2 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha2")

  # Objective definition
  m.setObjective(alpha2 * 100 + alpha1, GRB.MINIMIZE)

  # Constraints definition
  for i, j in sd_pairs:
    m.addConstr(
      quicksum(s_probability[q] * z1[i,j,q] * 1000 for q in range(num_scenario) if q not in perfect_scenario_set) >= (beta_low - prob_perfect) * 1000, "pass_beta1[%s,%s]" % (i,j)
    )

  for i, j in sd_pairs:
    m.addConstr(
      quicksum(s_probability[q] * z2[i,j,q] * 1000 for q in range(num_scenario) if q not in perfect_scenario_set) >= (beta_high - prob_perfect) * 1000, "pass_beta2[%s,%s]" % (i,j)
    )

  for q in range(num_scenario):
    if q in perfect_scenario_set:
      continue
    for e1, e2 in arcs:
      m.addConstr(
        quicksum(x1[q,l] for l in atraverse[e1, e2]) + quicksum(x2[q,l] for l in btraverse[e1, e2]) <= 1 * cap[e1, e2],
          "capacity[%s,%s,%s]" % (q, e1, e2)
      )

  for i, j in sd_pairs:
    for q in range(num_scenario):
      if q in perfect_scenario_set:
        continue
      m.addConstr(
        s1[i,j,q] >= t1[i,j,q],
        "loss_per_scenario1[%s,%s,%s]" % (i,j,q) 
      )
      m.addConstr(
        s2[i,j,q] >= t2[i,j,q],
        "loss_per_scenario2[%s,%s,%s]" % (i,j,q) 
      )

  for i, j in sd_pairs:
    for q in range(num_scenario):
      if q in perfect_scenario_set:
        continue
      m.addConstr(
        s1[i,j,q] <= alpha1 + 1 - z1[i,j,q],
        "bound_loss1[%s,%s,%s]" % (i,j,q) 
      )
      m.addConstr(
        s2[i,j,q] <= alpha2 + 1 - z2[i,j,q],
        "bound_loss2[%s,%s,%s]" % (i,j,q) 
      )

  for i, j in sd_pairs:
    for q in range(num_scenario):
      if q in perfect_scenario_set:
        continue
      m.addConstr(
        t1[i,j,q] == 1 - quicksum(x1[q,l] * y[q,l] for l in atunnels[i,j])/demand_low[i,j],
        "loss_per_flow_scenario1[%s,%s,%s]" % (i,j,q) 
      )
      m.addConstr(
        t2[i,j,q] == 1 - quicksum(x2[q,l] * y[q,l] for l in btunnels[i,j])/demand_high[i,j], 
        "loss_per_flow_scenario2[%s,%s,%s]" % (i,j,q) 
      )

  m.update()
  return m

def post_analysis(x, data, beta, demand, atunnels, priority):
  eps = 0.000000001
  sd_pairs = data['sd_pairs']
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
  logging.info("Post analysis for %s priority traffic:" % priority)
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
    if cur_avail >= beta and res > cur_loss:
      res = cur_loss
    logging.info("Loss level: %s%%, Availability: %s%%" % (cur_loss * 100, cur_avail * 100))
  logging.info("Loss level: %s%%, Availability: %s%%" % (100, 100))
  return res

def compute_pct_loss(base_model, data):
  eps = 0.0000001
  m = base_model.copy()

  sd_pairs = data['sd_pairs']
  num_scenario = data['num_scenario']
  anum_tunnel = data['anum_tunnel']
  s_probability = data['s_probability']
  demand_low = data['demand_low']
  demand_high = data['demand_high']
  failed_edge_set = data['failed_edge_set']
  atunnel_edge_set = data['atunnel_edge_set']
  atunnels = data['atunnels']
  btunnels = data['btunnels']
  no_failure_case = data['no_failure_case']
  
  y = {}

  for q in range(num_scenario):
    for l in range(anum_tunnel):
      y[q,l] = 1
      for u,v in failed_edge_set[q]:
        if (u,v) in atunnel_edge_set[l]:
          y[q,l] = 0
          break

  # Variable retrieval
  alpha1 = m.getVarByName("alpha1")
  alpha2 = m.getVarByName("alpha2")
  x1, x2, s1, t1, z1, s2, t2, z2 = {}, {}, {}, {}, {}, {}, {}, {}
  for q in range(num_scenario):
    for l in range(anum_tunnel):
      x1[q,l] = m.getVarByName("x1[%s,%s]" % (q,l))
      x2[q,l] = m.getVarByName("x2[%s,%s]" % (q,l))

  for q in range(num_scenario):
    for i, j in sd_pairs:
      s1[i,j,q] = m.getVarByName("s1[%s,%s,%s]" % (i,j,q))
      z1[i,j,q] = m.getVarByName("z1[%s,%s,%s]" % (i,j,q))
      t1[i,j,q] = m.getVarByName("t1[%s,%s,%s]" % (i,j,q))
      s2[i,j,q] = m.getVarByName("s2[%s,%s,%s]" % (i,j,q))
      z2[i,j,q] = m.getVarByName("z2[%s,%s,%s]" % (i,j,q))
      t2[i,j,q] = m.getVarByName("t2[%s,%s,%s]" % (i,j,q))

  # Solve
  #m.Params.nodemethod = 2
  #m.Params.MIPGap = 1e-6
  #m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()
 
  y1, y2 = {}, {}
  for q in range(num_scenario):
    y1[q], y2[q] = {}, {}
    for l in range(anum_tunnel):
      y1[q][l] = x1[q,l].X
      y2[q][l] = x2[q,l].X

  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    pct_loss_low = post_analysis(y1, data, data['beta_low'], demand_low, atunnels, 'low')
    pct_loss_high = post_analysis(y2, data, data['beta_high'], demand_high, btunnels, 'high')
    return pct_loss_low, pct_loss_high, m.Runtime
  return None, None, m.Runtime
