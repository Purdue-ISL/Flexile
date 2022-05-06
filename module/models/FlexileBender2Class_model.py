"""
FlexileBender2Class_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the FlexileBender2Class model in gurobi format.
"""

import sys
sys.path.append('/opt/gurobi/new/lib/python2.7/')
from gurobipy import *
import numpy as np
from collections import defaultdict
import logging
import time
import Online2Class_model

sys.path.append('../..')
import module.utils.utils as utils

def prepare_data(
    cap_file, tm_file, tm_index_low, tm_index_high, tunnel_file, scenario_file, scale_low, h_tunnel):

  # Index, Edge, Capacity definition
  split = 1
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
          for u in range(split):
            arcs.append((i, j, u))
            cap[i,j,u] = tcap / split
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
      for u in range(split):
        atraverse[i,j,u] = set()
        btraverse[i,j,u] = set()
      demand_low[i,j] = 0
      demand_high[i,j] = 0

  # Demand definition
  sd_pairs = []
  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = float(tm) * scale_low
        if int(h) == tm_index_low:
          demand_low[s,t] = tm
          if tm > 0.0:
            sd_pairs.append((s,t))

  with open(tm_file) as fin:
    for line in fin:
      if 's' not in line:
        s, t, h, tm = line.strip().split()
        tm = float(tm)
        if int(h) == tm_index_high:
          demand_high[s,t] = tm
          if tm > 0.0 and (s,t) not in sd_pairs:
            sd_pairs.append((s,t))
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
        # whether to consider for lantency-sensitive
        lt_sensitive = int(k) < h_tunnel
        edge_list = edge_list.split(',')
        for u in range(split):
          atunnel_edge_set[anum_tunnel] = set()
          atunnels[s,t].add(anum_tunnel)
          if lt_sensitive:
            btunnels[s,t].add(anum_tunnel)
          leng = 0
          for e in edge_list:
            e1, e2 = e.split('-')
            atunnel_edge_set[anum_tunnel].add((e1, e2, u))
            atraverse[e1,e2,u].add(anum_tunnel)
            if lt_sensitive:
              btraverse[e1,e2,u].add(anum_tunnel)
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
            if split > 1:
              u, v, n = e.split('-')
            else:
              u, v = e.split('-')
              n = 0
            failed_edge_set[num_scenario].add((u,v,int(n)))
        num_scenario = num_scenario + 1
  s_probability[num_scenario] = 1-total_prob
  failed_edge_set[num_scenario] = set()
  for e1, e2, n in arcs:
    failed_edge_set[num_scenario].add((e1, e2, n))
  num_scenario += 1

  # perfect Scenario definition
  perfect_scenario_set = set()
  num_perfect = 0
  prob_perfect = 0

  # All Scenario calculation
  all_probability = 0
  logging.info("all scenario_file: %s" % scenario_file)
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
          "prob_perfect": prob_perfect}

def post_analysis(x, data, beta, demand, tunnels, pri, z):
  eps = 0.0000001
  sd_pairs = data['sd_pairs']
  atunnel_edge_set = data['atunnel_edge_set']
  anum_tunnel = data['anum_tunnel']
  num_scenario = data['num_scenario']
  failed_edge_set = data['failed_edge_set']
  s_probability = data['s_probability']
  perfect_scenario_set = data['perfect_scenario_set']
  prob_perfect = data['prob_perfect']
  # Scenario definition
  y = {}
  loss = {}
  for i, j in sd_pairs:
    loss[i,j] = []
  for s in range(num_scenario):
    if s in perfect_scenario_set:
      for i, j in sd_pairs:
        if z[i,j,s,pri] > 0:
          loss[i,j].append((0, s_probability[s]))
      continue
    for l in range(anum_tunnel):
      y[l] = x[s,(l,pri)]
      for u,v,n in failed_edge_set[s]:
        if (u,v,n) in atunnel_edge_set[l]:
           y[l] = 0
           break
    for i, j in sd_pairs:
      flow = 0
      for l in tunnels[i,j]:
        flow = flow + y[l]
      loss[i,j].append((1 - flow/demand[i,j], s_probability[s]))
  logging.info("Post analysis for %s priority traffic:" % pri)
  loss_per_flow = {}
  scn = {}
  avail = {}
  for i, j in sd_pairs:
    loss[i,j].append((1.1, 1))
    loss[i,j].sort(key = lambda x:x[0])
    scn[i,j] = -1
    avail[i,j] = 0
  u = 1
  for loss_index in range(0, 100):
    cur_loss = loss_index * 0.01
    cur_avail = 1
    for i, j in sd_pairs:
      while loss[i,j][scn[i,j] + 1][0] <= cur_loss + eps:
        scn[i,j] = scn[i,j] + 1
        avail[i,j] = avail[i,j] + loss[i,j][scn[i,j]][1]
        if (i,j) not in loss_per_flow and avail[i,j] >= beta:
          loss_per_flow[i,j] = cur_loss
      if avail[i,j] < cur_avail:
        cur_avail = avail[i,j]
    if cur_avail > beta and cur_loss < u:
      u = cur_loss
    logging.info("Loss level: %s%%, Availability: %s%%" % (cur_loss * 100, cur_avail * 100))
  logging.info("Loss level: %s%%, Availability: %s%%" % (100, 100))
  return u

def benders_algorithm(beta_l, beta_h, data, step, output_routing_file):
  s_probability = data['s_probability']
  failed_edge_set = data['failed_edge_set']
  num_scenario = data['num_scenario']
  anum_tunnel = data['anum_tunnel']
  atunnel_edge_set = data['atunnel_edge_set']
  perfect_scenario_set = data['perfect_scenario_set']
  atunnels = data['atunnels']
  btunnels = data['btunnels']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  cap = data["capacity"]
  arcs = data["arcs"]
  
  eps = 0.000001
  scale_factor = 1

  y = {}
  z_ = {} 
  connection = {}
  discon = {}
  for q in range(num_scenario):
    for l in range(anum_tunnel):
      y[q,l] = 1
      for u,v,n in failed_edge_set[q]:
        if (u,v,n) in atunnel_edge_set[l]:
          y[q,l] = 0
          break
    discon[q, 'l'] = False
    discon[q, 'h'] = False
    for i,j in sd_pairs:
      total_connection = 0
      for l in atunnels[i,j]:
        total_connection += y[q,l]
      connection[q,i,j,'l'] = total_connection
      total_connection = 0
      for l in btunnels[i,j]:
        total_connection += y[q,l]
      connection[q,i,j,'h'] = total_connection
      z_[i,j,q,'l'] = 1
      z_[i,j,q,'h'] = 1
      if connection[q,i,j,'l'] == 0:
        discon[q, 'l'] = True
        z_[i,j,q,'l'] = 0
      if connection[q,i,j,'h'] == 0:
        discon[q, 'h'] = True
        z_[i,j,q,'h'] = 0

  beta = {}
  for i,j in sd_pairs:
    beta[i,j,'l'] = beta_l
    beta[i,j,'h'] = beta_h

  fe = {}
  for q in range(num_scenario):
    for u,v,n in arcs:
      if (u,v,n) in failed_edge_set[q]:
        fe[q,u,v,n] = 0
      else:
        fe[q,u,v,n] = 1

  last_rhs = {}
  var_x = {}
  model = smore_base_model(data, last_rhs, var_x)
  u_list = []
  v1_list = []
  v2_list = []
  w_list = []
  scenario_list = []
  optl_list = []
  opth_list = []
  total_rt = 0
  iter_limit = 5
  changed_set = range(num_scenario)
  tight_set = range(num_scenario)
  best_loss = 101
  loss_per_scen = {}
  for iteration in range(iter_limit):
    lower_bound = 0
    x = {}
    max_rt = 0
    scenario_count = 0
    resolve = []
    dual_var_list = []
    for q in range(num_scenario):
      if q in perfect_scenario_set or q not in changed_set:
        logging.info("skipping scenario %s for iteration %s" % (q, iteration))
        for l in range(anum_tunnel):
          for pri in ['l', 'h']:
            x[q,(l,pri)] = x_[q,(l,pri)]
        continue
      resolve.append(q)
      scenario_count += 1
      logging.info("computing scenario %s for iteration %s" % (q, iteration))
      x_q, u_k, v1_k, v2_k, w_k, obj, theta_l, theta_h, rt = solve_smore(model, data, q, failed_edge_set[q], z_, last_rhs, var_x)   
      loss_per_scen[q] = theta_l + 0.0000001
      logging.info("failed set: %s" % failed_edge_set[q])
      if iteration == 0:      
        optl_list.append(theta_l)
        opth_list.append(theta_h)
      if obj < eps and iteration == 0:
        perfect_scenario_set.add(q)
        for i,j in sd_pairs:
          for pri in ['l', 'h']:
            if z_[i,j,q,pri] > 0:
              beta[i,j,pri] -= s_probability[q]
      else:
        dual_var_list.append((u_k, v1_k, v2_k, w_k))
        u_list.append(u_k)
        v1_list.append(v1_k)
        v2_list.append(v2_k)
        w_list.append(w_k)
        scenario_list.append(q)
      x.update(x_q)
      lower_bound = max(lower_bound, obj)
      max_rt = max(max_rt, rt)
    logging.info("computing chosen cuts...")
    tmptime = time.time()
    if iteration == 0:
      data['perfect_scenario_set'] = perfect_scenario_set
      logging.info("number of perfect scenarios: %s", len(perfect_scenario_set))
    total_rt += rt

    u1 = post_analysis(x, data, beta_l, data['demand_low'], data['atunnels'], 'l', z_)
    u2 = post_analysis(x, data, beta_h, data['demand_high'], data['btunnels'], 'h', z_)
    if u2 * 100 + u1 < best_loss:
      best_loss = u2 * 100 + u1
    logging.info("iteration %s best_loss = %s, runtime = %s." % (iteration, best_loss, total_rt))
    logging.info("computed scenario number: %s", scenario_count)
    x_ = x
    last_changed_set = changed_set
    if iteration != iter_limit - 1:
      z_, upper_bound, rt, changed_set, tight_set = solve_benders(beta, data, u_list, v1_list, v2_list, w_list, scenario_list, z_, scale_factor, step, connection, fe)
      for q in last_changed_set:
        if q not in resolve and q not in changed_set:
          changed_set.append(q)
      total_rt += rt
    else:
      l_list, h_list, t_list = [],[],[]
      loss_data, x1, x2 = {}, {}, {}
      for q in range(num_scenario):
        #loss_l, loss_h = worst_loss(data, x, z_, q)
        x1[q], x2[q], t, loss_l, loss_h, loss_l_list = Online2Class_model.get_online_routing(data, z_, q, loss_per_scen[q])
        loss_data[q] = loss_l_list
        l_list.append(loss_l)
        h_list.append(loss_h)
      utils.output_routing_2class(output_routing_file, num_scenario, x1, x2)
    logging.info("iteration %s lb = %s, ub = %s." % (iteration, lower_bound, upper_bound))
  logging.info("best loss: %s" % best_loss)
  return u1, u2, total_rt

def difference(store_z, z_, data):
  num_scenario = data['num_scenario']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  perfect_scenario_set = data['perfect_scenario_set']
  total_changed_scenario = 0
  changed_set = []
  for q in range(num_scenario):
    if q not in perfect_scenario_set:
      for pri in ['l', 'h']:
        for i, j in sd_pairs:
          if store_z[i,j,q,pri] != z_[i,j,q,pri]:
            changed_set.append(q)
            total_changed_scenario += 1
            break
  logging.info("total changed scenarios: %s" % total_changed_scenario)
  return changed_set

def solve_benders(beta, data, u_list, v1_list, v2_list, w_list, scenario_list, z_, scale_factor, step, connection, fe):
  nodes, arcs = data['nodes'], data['arcs']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  cap = data['capacity']
  num_scenario = data['num_scenario']
  s_probability = data['s_probability']
  perfect_scenario_set = data['perfect_scenario_set']

  m = Model('benders')

  store_z = {}

  # Variable definition
  z = {}

  for q in range(num_scenario):
    if q not in perfect_scenario_set:
      for i, j in sd_pairs:
        for pri in ['l', 'h']:
          z[i,j,q,pri] = m.addVar(lb=0.0, vtype=GRB.BINARY, name="z[%s,%s,%s,%s]" % (i,j,q,pri))

  alpha = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")

  # Objective definition
  m.setObjective(alpha, GRB.MINIMIZE)

  # Constraints definition
  count = 0
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      m.addConstr(
        quicksum(z[i,j,q,pri] * s_probability[q] * 10000 for q in range(num_scenario-1) if q not in perfect_scenario_set) >= beta[i,j,pri] * 10000,
          "enough_prob[%s,%s,%s]" % (i, j, pri) 
      )
    for q in range(num_scenario):
      for pri in ['l', 'h']:
        if q not in perfect_scenario_set and connection[q,i,j,pri] == 0:
          m.addConstr(z[i,j,q,pri] == 0,
            "bad_scenario[%s,%s,%s,%s]" % (i, j, q, pri) 
          )

  for b in range(len(u_list)):
    u = u_list[b]
    v1 = v1_list[b]
    v2 = v2_list[b]
    w = w_list[b]
    q = scenario_list[b]
    m.addConstr(
          quicksum(cap[e1,e2,n] * u[e1,e2,n] * fe[q,e1,e2,n] for e1,e2,n in arcs)
          + quicksum(v1[i,j] + v2[i,j] for i,j in sd_pairs)
          + quicksum((z[i,j,q,pri]-1) * w[i,j,pri] for i,j in sd_pairs for pri in ['l','h']) <= alpha,
          "alpha_constr[%s,%s]" % (b, q) 
    )

  if step > 0:
    lhs, count_1 = {}, {}
    lhs['l'] = 0
    lhs['h'] = 0
    count_1['l'] = 0
    count_1['h'] = 0
    for q in range(num_scenario-1):
      if q not in perfect_scenario_set:
        for i, j in sd_pairs:
          for pri in ['l', 'h']:
            if z_[i,j,q,pri] == 1:
              count_1[pri] += 1
              lhs[pri] = lhs[pri] - z[i,j,q,pri]
            else:
              lhs[pri] = lhs[pri] + z[i,j,q,pri]
    for pri in ['l', 'h']:
      m.addConstr(
        count_1[pri] + lhs[pri] <= step, "diff_bound[%s]" % (pri) 
      )

  # Solve
  #m.Params.nodemethod = 2
  #m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)

  eps = 0.000000001

  tight_set = set()

  store_z = {}
  for q in range(num_scenario):
    for i, j in sd_pairs:  
      for pri in ['l', 'h']:
        if q not in perfect_scenario_set:
          store_z[i,j,q,pri] = z[i,j,q,pri].X
        else:
          store_z[i,j,q,pri] = z_[i,j,q,pri]
  changed_set = difference(store_z, z_, data)
  logging.info("number of changed scenarios: %s" % len(changed_set))

  total_diff = 0
  for q in range(num_scenario-1):
    if q not in perfect_scenario_set: 
      for pri in ['l', 'h']:
        for i, j in sd_pairs:
          if z_[i,j,q,pri] == 1:
            total_diff += z_[i,j,q,pri] - z[i,j,q,pri].X
          else:
            total_diff += z[i,j,q,pri].X
  logging.info("total diff: %s" % total_diff)

  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return store_z, m.ObjVal, m.Runtime, changed_set, tight_set
  return None, None

def smore_base_model(data, last_rhs, x):
  arcs = data['arcs']
  sd_pairs = data['sd_pairs']
  cap, demand_low, demand_high = data['capacity'], data['demand_low'], data['demand_high']
  atunnels = data['atunnels']
  btunnels = data['btunnels']
  atraverse = data['atraverse']
  btraverse = data['btraverse']
  anum_tunnel = data['anum_tunnel']

  m = Model('smore')
  # Variable definition
  t, theta = {}, {}
  for l in range(anum_tunnel):
    for pri in ['l', 'h']:
      x[l,pri] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1[%s,%s]" % (l,pri))
  
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      t[i,j,pri] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t[%s,%s,%s]" % (i,j,pri))

  theta['l'] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta[l]")
  theta['h'] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta[h]")

  # Objective definition
  m.setObjective(theta['h'] * 100 + theta['l'], GRB.MINIMIZE)

  # Constraints definition
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      m.addConstr(
        theta[pri] - t[i,j,pri] >= 0,
        "theta[%s,%s,%s]" % (i,j,pri) 
      )
      last_rhs[i,j,pri] = 0

  for e1, e2, n in arcs:
    m.addConstr(
      quicksum(x[l,pri] for l in atraverse[e1, e2, n] for pri in ['l', 'h']) <= cap[e1, e2, n],
        "capacity[%s,%s,%s]" % (e1, e2, n)
    )
    last_rhs[e1, e2, n] = cap[e1, e2, n]

  for i, j in sd_pairs:
    m.addConstr(
      quicksum(x[l,'l']/demand_low[i,j] for l in atunnels[i,j]) >= 1 - t[i,j,'l'],
      "demand_low[%s,%s]" % (i,j) 
    )  
    m.addConstr(
      quicksum(x[l,'h']/demand_high[i,j] for l in btunnels[i,j]) >= 1 - t[i,j,'h'],
      "demand_high[%s,%s]" % (i,j) 
    )  
  # Solve
  #m.Params.nodemethod = 2
  #m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 1
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)
  return m

def solve_smore(m, data, q, failed_edge_set, z_, last_rhs, x):
  # Parameters
  arcs = data['arcs']
  sd_pairs = data['sd_pairs']
  cap, demand_low, demand_high = data['capacity'], data['demand_low'], data['demand_high']
  anum_tunnel = data['anum_tunnel']
  atunnel_edge_set = data['atunnel_edge_set']

  # Constraints definition
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      if last_rhs[i,j,pri] != z_[i,j,q,pri] - 1:
        constr = m.getConstrByName("theta[%s,%s,%s]" % (i,j,pri))
        constr.setAttr("rhs", z_[i,j,q,pri] - 1)
        last_rhs[i,j,pri] = z_[i,j,q,pri] - 1

  for e1, e2, n in arcs:
    capacity = cap[e1, e2, n]
    if (e1, e2, n) in failed_edge_set:
      capacity = 0
    if last_rhs[e1, e2, n] != capacity:
      constr = m.getConstrByName("capacity[%s,%s,%s]" % (e1, e2, n))
      constr.setAttr("rhs", capacity)
      last_rhs[e1, e2, n] = capacity

  theta_l = m.getVarByName("theta[l]")
  theta_h = m.getVarByName("theta[h]")

  # Solve
  #m.update()
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)

  eps = 0.0000001
  store_x = {}
  for i in x:
    store_x[q,i] = x[i].X

  store_u = {}
  store_v1 = {}
  store_v2 = {}
  store_w = {}
  theta_n = {}
  capa_n = {}
  demand_low_n = {}
  demand_high_n = {}
  for i, j in sd_pairs:
    for pri in ['l', 'h']:
      name = "theta[%s,%s,%s]" % (i,j,pri) 
      theta_n[name] = (i,j,pri)
    name = "demand_low[%s,%s]" % (i,j)
    demand_low_n[name] = (i,j)
    name = "demand_high[%s,%s]" % (i,j)
    demand_high_n[name] = (i,j)
  for e1, e2, n in arcs:
    name = "capacity[%s,%s,%s]" % (e1, e2, n)
    capa_n[name] = (e1, e2, n)
  constraints = m.getConstrs()
  check_dual = 0
  for constr in constraints:
    name = constr.constrName
    if name in theta_n:
      store_w[theta_n[name][0], theta_n[name][1], theta_n[name][2]] = constr.pi
      check_dual += constr.pi * (z_[theta_n[name][0], theta_n[name][1], q, theta_n[name][2]] - 1)
    if name in capa_n:
      store_u[capa_n[name][0], capa_n[name][1], capa_n[name][2]] = constr.pi
      if (capa_n[name][0], capa_n[name][1], capa_n[name][2]) not in failed_edge_set:
        check_dual += constr.pi * cap[capa_n[name][0], capa_n[name][1], capa_n[name][2]]  
    if name in demand_low_n:
      store_v1[demand_low_n[name][0], demand_low_n[name][1]] = constr.pi
      check_dual += constr.pi
    if name in demand_high_n:
      store_v2[demand_high_n[name][0], demand_high_n[name][1]] = constr.pi
      check_dual += constr.pi
    
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return store_x, store_u, store_v1, store_v2, store_w, m.ObjVal, theta_l.X, theta_h.X, m.Runtime
  return None, None, m.Runtime
