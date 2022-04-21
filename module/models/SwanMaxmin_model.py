"""
SwanMaxmin_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the SWAN-Maxmin model in gurobi format.
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
  tflist = []
  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = float(tm)
        if int(h) == tm_index_low:
          demand_low[s,t] = tm * scale_low
          if tm > 0.0:
            tflist.append(tm)
            sd_pairs.append((s,t))
  rate1 = min(tflist) * 0.95
  maxd1 = max(tflist)

  tflist = []
  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = float(tm)
        if int(h) == tm_index_high:
          demand_high[s,t] = tm
          if tm > 0.0:
            tflist.append(tm) 
            if (s,t) not in sd_pairs:
              sd_pairs.append((s,t))
  rate2 = min(tflist) * 0.95
  maxd2 = max(tflist)
  sd_pairs_set = set(sd_pairs)

  # Tunnel node and edge set definition
  atunnel_edge_set = {}
  atunnel_length = {}
  logging.debug("tunnel_file: %s" % tunnel_file)
  anum_tunnel = 0
  with open(tunnel_file) as fin:
    for line in fin:
      if 's' not in line:
        unpack = line.strip().split()
        s, t, k, edge_list, leng = unpack
        leng = float(leng)
        #s, t, k, edge_list = unpack
        # whether to consider for lantency-sensitive
        lt_sensitive = int(k) < 3
        edge_list = edge_list.split(',')
        atunnel_edge_set[anum_tunnel] = set()
        atunnels[s,t].add(anum_tunnel)
        if lt_sensitive:
            btunnels[s,t].add(anum_tunnel)
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
        num_scenario = num_scenario + 1
  s_probability[num_scenario] = 1-total_prob
  failed_edge_set[num_scenario] = set()
  for e1, e2 in arcs:
    failed_edge_set[num_scenario].add((e1, e2))

  # All Scenario calculation
  all_probability = 0
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
          "rate1": rate1, "rate2": rate2, "maxd1": maxd1, "maxd2": maxd2,
          "alpha": 2}

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

def compute_pct_loss(data, beta1, beta2):
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  all_probability = data['all_probability']
  demand_low = data['demand_low']
  demand_high = data['demand_high']
  cap = data['capacity']
  atunnels = data['atunnels']
  btunnels = data['btunnels']
  atraverse = data['atraverse']
  btraverse = data['btraverse']
  sd_pairs = data['sd_pairs']
  x1 = {}
  x2 = {}
  u1 = data['rate1']
  maxd1 = data['maxd1']
  u2 = data['rate2']
  maxd2 = data['maxd2']
  prob = 0
  total_rt = 0
  for i in range(num_scenario):
    logging.info("Solving scenario: %f" % i)
    tmp_cap = {}
    for c in cap:
      tmp_cap[c] = cap[c] * 1
    x2[i], rt = solve_iterate(data, failed_edge_set[i], tmp_cap, demand_high, btunnels, atraverse, u2, maxd2)
    total_rt += rt
    saturate_links = {}
    for c in tmp_cap:
      for l in atraverse[c]:
        tmp_cap[c] -= x2[i][l]
        if tmp_cap[c] < 0.000001:
          saturate_links[c] = 1
    x1[i], rt = solve_iterate(data, failed_edge_set[i], tmp_cap, demand_low, atunnels, atraverse, u1, maxd1)
    total_rt += rt
    for c in tmp_cap:
      for l in atraverse[c]:
        tmp_cap[c] -= x1[i][l]
      #print "cap:", c, tmp_cap[c]
    connection = 0
    if i > 0:
      for s, t in sd_pairs:
        tmp_flow = 0
        for l in atunnels[s,t]:
          tmp_flow += x1[i][l]
  pct_loss_low = post_analysis(x1, data, beta1, demand_low, atunnels, 'low')
  pct_loss_high = post_analysis(x2, data, beta2, demand_high, btunnels, 'high')
  return pct_loss_low, pct_loss_high, total_rt

def solve_iterate(data, failed_edges, cap, demand, atunnels, atraverse, u, maxd):
  alpha = data['alpha']
  T = int(math.log(maxd / u, alpha)) + 1
  logging.info("Total iterations: %s" % T)
  last_low = u
  flow_set = {}
  eps = 0.000000001
  x, obj, m, total_rt = solve_once_base(data, failed_edges, cap, demand, atunnels, atraverse, 0, u, flow_set)
  for k in range(T):
    x, obj, rt = solve_once_tp(m, data, failed_edges, cap, demand, atunnels, atraverse, last_low, last_low * alpha, flow_set)
    total_rt += rt
    if obj < eps:
      break
    last_low = last_low * alpha
    logging.info("iteration: %s, obj: %s" % (k, obj))
  return x, total_rt

def solve_once_tp(m, data, failed_edges, cap, demand, atunnels, atraverse, b_low, b_high, flow_set):
  # Parameters
  nodes, arcs = data['nodes'], data['arcs']
  nodes_set, arcs_set = \
      data['nodes_set'], data['arcs_set']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']

  x, b = {}, {}
  for l in range(anum_tunnel):
    x[l] = m.getVarByName("x[%s]" % (l))
  for i, j in sd_pairs:
    b[i,j] = m.getVarByName("b[%s,%s]" % (i,j))

  for i, j in sd_pairs:
    if (i,j) in flow_set:
      cur_low = flow_set[i,j]
      cur_high = flow_set[i,j]
    else:
      cur_low = b_low
      cur_high = min(b_high, demand[i,j])
    constr = m.getConstrByName("b_low[%s,%s]" % (i,j))
    constr.setAttr("rhs", cur_low)
    constr = m.getConstrByName("b_high[%s,%s]" % (i,j))
    constr.setAttr("rhs", cur_high)

  # Solve
  m.optimize()

  eps = 0.000000001
  y = {}
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    for i in x:
      y[i] = x[i].X
    for i,j in sd_pairs:
      if (i,j) not in flow_set:
        if min(b_high, demand[i,j]) - b[i,j].X > eps:
          flow_set[i,j] = b[i,j].X 
        if demand[i,j] - b[i,j].X < eps:
          flow_set[i,j] = b[i,j].X 
    return y, m.ObjVal, m.Runtime
  else:
    logging.info("no solution!")
    for i in x:
      y[i] = 0
    return y, 0, m.Runtime

def solve_once_base(data, failed_edges, cap, demand, atunnels, atraverse, b_low, b_high, flow_set):
  nodes, arcs = data['nodes'], data['arcs']
  nodes_set, arcs_set = \
      data['nodes_set'], data['arcs_set']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']

  eps = 0.0000001

  valid_tunnel = {}
  for i, j in sd_pairs:
    for l in atunnels[i,j]:
      valid_tunnel[l] = 0

  # Gurobi Solver Model
  m = Model('mcf')

  # Variable definition
  x, b = {}, {}
  for l in range(anum_tunnel):
    x[l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x[%s]" % (l))
  for i, j in sd_pairs:
    b[i,j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="b[%s,%s]" % (i,j))
 
  # Objective definition
  m.setObjective(quicksum(b[i,j] for i,j in sd_pairs), GRB.MAXIMIZE)

  # Constraints definition
  for e1, e2 in arcs:
    c = 1 #0.95
    if (e1, e2) in failed_edges:
      c = 0
    m.addConstr(
      quicksum(x[l] for l in atraverse[e1, e2]) <= cap[e1, e2] * c,
        "capacity[%s,%s]" % (e1, e2)
    )

  for l in range(anum_tunnel):
    if l in valid_tunnel:
      m.addConstr(
        x[l] >= 0,
        "x[%s]" % (l) 
      )
    else:
      m.addConstr(
        x[l] == 0,
        "x[%s]" % (l) 
      )
  
  for i, j in sd_pairs:
    m.addConstr(
      quicksum(x[l] for l in atunnels[i,j]) == b[i,j],
      "b[%s,%s]" % (i,j) 
    )
    if (i,j) in flow_set:
      cur_low = flow_set[i,j]
      cur_high = flow_set[i,j]
    else:
      cur_low = b_low
      cur_high = min(b_high, demand[i,j])
    m.addConstr(
      b[i,j] >= cur_low,
      "b_low[%s,%s]" % (i,j) 
    )
    m.addConstr(
      b[i,j] <= cur_high,
      "b_high[%s,%s]" % (i,j) 
    )
  # Solve
  #m.Params.nodemethod = 2
  m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)
  y = {}
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    for i in x:
      y[i] = x[i].X
    for i,j in sd_pairs:
      if (i,j) not in flow_set:
        if min(b_high, demand[i,j]) - b[i,j].X > eps:
          flow_set[i,j] = b[i,j].X 
        if demand[i,j] - b[i,j].X < eps:
          flow_set[i,j] = b[i,j].X 
    return y, m.ObjVal, m, m.Runtime
  else:
    logging.info("no solution!")
    for i in x:
      y[i] = 0
    return y, 0, m, m.Runtime
