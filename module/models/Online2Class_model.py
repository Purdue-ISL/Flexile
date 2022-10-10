"""
Online2Class_model.py
Run with python2.7+ (but not python3)
This file provides a function to generate online routing through critical scenarios.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging
import time

def get_online_routing(data, z, q, loss):  
  logging.info("generating online routing...")  
  failed_edge_set = data['failed_edge_set']
  demand1 = data['demand_low']
  demand2 = data['demand_high']
  cap = data['capacity']
  atunnels = data['atunnels']
  btunnels = data['btunnels']
  atraverse = data['atraverse']
  btraverse = data['btraverse']
  sd_pairs = data['sd_pairs']
  flow_set = {}
  t = 0
  for i,j in sd_pairs:
    if z[i,j,q, 'l'] > 0.5:
      flow_set['l',i,j] = (demand1[i,j] * (1 - loss), 'fixed')
    if z[i,j,q,'h'] > 0.5:
      flow_set['h',i,j] = (demand2[i,j], 'fixed')
  _, _, rt = solve_iterate(data, failed_edge_set[q], cap, demand2, flow_set, 'h', loss)
  t += rt
  for i,j in sd_pairs:
    if z[i,j,q,'l'] > 0.5:
      flow_set['l',i,j] = (demand1[i,j] * (1 - loss), 'lb')
  x1, x2, rt = solve_iterate(data, failed_edge_set[q], cap, demand1, flow_set, 'l', loss)
  t += rt
  loss_l, loss_h = 0, 0
  loss_l_list = []
  for i,j in sd_pairs:
    connect = 0
    d = demand1[i,j]
    flow = 0
    for l in atunnels[i,j]:
      flag = 1
      for e1, e2, u in failed_edge_set[q]:
        if l in atraverse[e1, e2, u]:
          flag = 0
          break
      if flag == 1:
        connect += 1
        flow += x1[l]
    if connect > 0:
      loss_l = max(1 - flow/d, loss_l)
      loss_l_list.append((1 - flow/d, i, j))
  loss_l_list.sort(reverse=True)
  if len(loss_l_list) > 0 and loss_l_list[0][0] > 0.0000001:
    print "non-zero case:", q, [e[0] for e in loss_l_list]
  for i,j in sd_pairs:
    connect = 0
    d = demand2[i,j]
    flow = 0
    for l in btunnels[i,j]:
      flag = 1
      for e1, e2, u in failed_edge_set[q]:
        if l in atraverse[e1, e2, u]:
          flag = 0
          break
      if flag == 1:
        connect += 1
        flow += x2[l]
    if connect > 0:
      loss_h = max(1 - flow/d, loss_h)
  return x1, x2, t, loss_l, loss_h, loss_l_list

def solve_iterate(data, failed_edges, cap, demand, flow_set, pri, loss):
  fraction = 0.1
  eps = 0.000001
  t = 0
  lastobj = 0
  for k in range(int(1 / fraction)):
    x, x_, obj, rt = solve_once_loss(data, failed_edges, cap, demand, k * fraction, (k + 1) * fraction, flow_set, pri, loss)
    t += rt
    if obj <= lastobj + eps:
      break
    lastobj = obj
    logging.info("iteration: %s, obj: %s" % (k, obj))
  return x, x_, t

def solve_once_loss( data, failed_edges, cap, demand, b_low, b_high, flow_set, cur_pri, loss):
  # Parameters
  nodes, arcs = data['nodes'], data['arcs']
  nodes_set, arcs_set = \
      data['nodes_set'], data['arcs_set']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  btunnels = data['btunnels']
  atunnels = data['atunnels']
  atraverse = data['atraverse']

  # Gurobi Solver Model
  m = Model('teavar')

  # Variable definition
  x, b = {}, {}
  for l in range(anum_tunnel):
    for pri in ['l', 'h']:
      x[pri, l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x%s[%s]" % (pri,l))
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      b[pri, i,j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="b%s[%s,%s]" % (pri,i,j))
 
  # Objective definition
  m.setObjective(quicksum(b[pri,i,j] for pri in ['l','h'] for i,j in sd_pairs), GRB.MAXIMIZE)

  # Constraints definition
  for e1, e2, u in arcs:
    c = 1 #0.95
    if (e1, e2, u) in failed_edges:
      c = 0
    m.addConstr(
      quicksum(x['h', l] + x['l', l] for l in atraverse[e1, e2, u]) <= cap[e1, e2, u] * c,
        "capacity[%s,%s, %s]" % (e1, e2, u)
    )

  for l in range(anum_tunnel):
    for pri in ['l', 'h']:
      m.addConstr(
        x[pri, l] >= 0,
        "x%s[%s]" % (pri,l) 
      )
  
  for i, j in sd_pairs:
    m.addConstr(
      quicksum(x['h', l] for l in btunnels[i,j]) == b['h',i,j],
        "bh[%s,%s]" % (i,j) 
    )
    m.addConstr(
      quicksum(x['l', l] for l in atunnels[i,j]) == b['l',i,j],
        "bl[%s,%s]" % (i,j) 
    )
    for pri in ['h', 'l']:
      if (pri,i,j) in flow_set:
        if flow_set[pri,i,j][1] == 'fixed':
          m.addConstr(
            b[pri,i,j] == flow_set[pri,i,j][0],
            "flow_set%s[%s,%s]" % (pri,i,j) 
          )
        if flow_set[pri,i,j][1] == 'lb':
          m.addConstr(
            b[pri,i,j] >= flow_set[pri,i,j][0],
            "flow_setlb%s[%s,%s]" % (pri,i,j) 
          )
          m.addConstr(
            b[pri,i,j] <= demand[i,j],
            "flow_setub%s[%s,%s]" % (pri,i,j) 
          )
      else:
        if pri != cur_pri:
          m.addConstr(
            b[pri,i,j] == 0,
            "bno[%s,%s]" % (i,j) 
          )
        else:
          m.addConstr(
            b[pri,i,j] >= b_low * demand[i,j],
            "b_low[%s,%s]" % (i,j) 
          )
          m.addConstr(
            b[pri,i,j] <= b_high * demand[i,j],
            "b_high[%s,%s]" % (i,j) 
          )

  # Solve
  #m.Params.nodemethod = 2
  m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 1
  m.optimize()

  eps = 0.000000001
  logging.info("Runtime: %f seconds" % m.Runtime)
  y, y_ = {}, {}
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    for l in range(anum_tunnel):
      y[l] = x['l',l].X
      y_[l] = x['h',l].X
    for i,j in sd_pairs:
      if (cur_pri,i,j) not in flow_set:
        if b_high * demand[i,j] - b[cur_pri,i,j].X > eps:
          flow_set[cur_pri,i,j] = (b[cur_pri,i,j].X, 'fixed') 
        if demand[i,j] - b[cur_pri,i,j].X < eps:
          flow_set[cur_pri,i,j] = (b[cur_pri,i,j].X, 'fixed') 
    logging.info("failed edges: %s" % failed_edges)
    logging.info("flow_set size: %s" % len(flow_set))
    return y, y_, m.ObjVal, m.Runtime
  else:
    logging.info("no solution!")
    for i in x:
      y[i] = 0
      y_[i] = 0
    return y, y_, 0, m.Runtime
