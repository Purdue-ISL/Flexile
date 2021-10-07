"""
FlexileBender_model.py
Run with python2.7+ (but not python3)
This file reads topology, traffic matrix, tunnel, and implements the Flexile Bender model in gurobi format.
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

  perfect_scenario_set = set()
  prob_perfect = 0

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
          "s_probability": s_probability,
          "perfect_scenario_set": perfect_scenario_set, "prob_perfect": prob_perfect}

def post_analysis(x, data, beta, z, is_final):
  eps = 0.0000001
  sd_pairs = data['sd_pairs']
  demand = data['demand']
  atunnels = data['atunnels']
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
        if z[i,j,s] > 0:
          loss[i,j].append((0, s_probability[s]))
      continue

    for l in range(anum_tunnel):
      y[l] = x[s,l]
      for u,v in failed_edge_set[s]:
        if (u,v) in atunnel_edge_set[l]:
           y[l] = 0
           break
        
    for i, j in sd_pairs:
      flow = 0
      for l in atunnels[i,j]:
        flow = flow + y[l]
      loss[i,j].append((1 - flow/demand[i,j], s_probability[s]))
  logging.info("Post analysis:")
  loss_per_flow = {}
  scn = {}
  avail = {}
  for i, j in sd_pairs:
    loss[i,j].append((1.1, 1))
    loss[i,j].sort(key = lambda x:x[0])
    scn[i,j] = -1
    avail[i,j] = 0
  u = 1
  res = 1
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
    if cur_avail + eps >= beta and res > cur_loss:
      res = cur_loss
    if is_final:
      logging.info("Loss level: %s%%, Availability: %s%%" % (cur_loss * 100, cur_avail * 100))
  if is_final:
    logging.info("Loss level: %s%%, Availability: %s%%" % (100, 100))
  return res

def benders_algorithm(beta_, data, scenario_file, step):
  s_probability = data['s_probability']
  failed_edge_set = data['failed_edge_set']
  num_scenario = data['num_scenario']
  anum_tunnel = data['anum_tunnel']
  atunnel_edge_set = data['atunnel_edge_set']
  perfect_scenario_set = data['perfect_scenario_set']
  prob_perfect = data['prob_perfect']
  atunnels = data['atunnels']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  arcs = data["arcs"]
  
  eps = 0.000001
  scale_factor = 1

  beta = {}
  for i,j in sd_pairs:
    beta[i,j] = beta_

  y = {}
  z_ = {}
  connection = {}
  discon = {}
  for q in range(num_scenario):
    for l in range(anum_tunnel):
      y[q,l] = 1
      for u,v in failed_edge_set[q]:
        if (u,v) in atunnel_edge_set[l]:
          y[q,l] = 0
          break
    discon[q] = False
    for i,j in sd_pairs:
      total_connection = 0
      for l in atunnels[i,j]:
        total_connection += y[q,l]
      connection[q,i,j] = total_connection
      z_[i,j,q] = 1
      if connection[q,i,j] == 0:
        discon[q] = True
        z_[i,j,q] = 0
        
  fe = {}
  for q in range(num_scenario):
    for u,v in arcs:
      if (u,v) in failed_edge_set[q]:
        fe[q,u,v] = 0
      else:
        fe[q,u,v] = 1

  last_rhs = {}
  var_x = {}
  model = smore_base_model(data, last_rhs, var_x)
  u_list = []
  v_list = []
  w_list = []
  scenario_list = []
  total_rt = 0
  iter_limit = 5
  changed_set = range(num_scenario)
  best_loss = 1
  for iteration in range(iter_limit):
    lower_bound = 0
    x = {}
    max_rt = 0
    scenario_count = 0
    resolve = []
    for q in range(num_scenario):
      if q in perfect_scenario_set or q not in changed_set:
        logging.info("skipping scenario %s for iteration %s" % (q, iteration))
        for l in range(anum_tunnel):
          x[q,l] = x_[q,l]
        continue
      resolve.append(q)
      scenario_count += 1
      logging.info("computing scenario %s for iteration %s" % (q, iteration))
      x_q, u_k, v_k, w_k, obj, rt = solve_smore(model, data, q, failed_edge_set[q], z_, last_rhs, var_x)

      total_rt += rt
      if obj < eps and iteration == 0:
        perfect_scenario_set.add(q)
        prob_perfect += s_probability[q]
        for i,j in sd_pairs:
          if z_[i,j,q] > 0:
            beta[i,j] -= s_probability[q]
      else:
        u_list.append(u_k)
        v_list.append(v_k)
        w_list.append(w_k)
        scenario_list.append(q)
      x.update(x_q)
      lower_bound = max(lower_bound, obj)
      max_rt = max(max_rt, rt)
    if iteration == 0:
      data['prob_perfect'] = prob_perfect
      data['perfect_scenario_set'] = perfect_scenario_set
      logging.info("number of perfect scenarios: %s", len(perfect_scenario_set))
      logging.info("perfect prob: %s", prob_perfect)
    is_final = (iteration == iter_limit - 1)
    u = post_analysis(x, data, beta_, z_, is_final)
    if u < best_loss:
      best_loss = u
    logging.info("iteration %s best_loss = %s." % (iteration, best_loss))
    logging.info("computed scenario number: %s", scenario_count)
    if iteration < iter_limit - 1:
      x_ = x
      last_changed_set = changed_set
      z_, upper_bound, rt, changed_se = solve_benders(beta, data, u_list, v_list, w_list, scenario_list, z_, scale_factor, step, True, connection, fe)
      for q in last_changed_set:
        if q not in resolve and q not in changed_set:
          changed_set.append(q)
      total_rt += rt
      logging.info("iteration %s lb = %s, ub = %s." % (iteration, lower_bound, upper_bound))
  logging.info("best loss: %s" % best_loss)
  return best_loss, total_rt

def difference(store_z, z_, data):
  num_scenario = data['num_scenario']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  perfect_scenario_set = data['perfect_scenario_set']
  total_changed_scenario = 0
  changed_set = []
  for q in range(num_scenario):
    if q not in perfect_scenario_set:
      for i, j in sd_pairs:
        if store_z[i,j,q] != z_[i,j,q]:
          changed_set.append(q)
          total_changed_scenario += 1
          break
  logging.info("total changed scenarios: %s" % total_changed_scenario)
  return changed_set

def solve_benders(beta, data, u_list, v_list, w_list, scenario_list, z_, scale_factor, step, is_ip, connection, fe):
  nodes, arcs = data['nodes'], data['arcs']
  sd_pairs, node_pairs = data['sd_pairs'], data['node_pairs']
  cap, demand = data['capacity'], data['demand']
  num_scenario = data['num_scenario']
  s_probability = data['s_probability']
  perfect_scenario_set = data['perfect_scenario_set']
  prob_perfect = data['prob_perfect']

  m = Model('benders')

  logging.info("prob perfect in master: %s" % prob_perfect)

  store_z = {}

  # Variable definition
  z = {}

  for q in range(num_scenario):
    if q not in perfect_scenario_set:
      for i, j in sd_pairs:
        if is_ip:
          z[i,j,q] = m.addVar(lb=0.0, vtype=GRB.BINARY, name="z[%s,%s,%s]" % (i,j,q))
        else:
          z[i,j,q] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z[%s,%s,%s]" % (i,j,q))

  alpha = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")

  # Objective definition
  m.setObjective(alpha, GRB.MINIMIZE)

  # Constraints definition
  count = 0
  for i, j in sd_pairs:
    m.addConstr(
      quicksum(z[i,j,q] * s_probability[q] * 100000 for q in range(num_scenario-1) if q not in perfect_scenario_set) >= beta[i,j] * 100000,
        "enough_prob[%s,%s]" % (i, j) 
    )
    #m.addConstr(z[i,j,num_scenario-1] == 0,
    #    "bad_scenario[%s,%s]" % (i, j) 
    #)
    for q in range(num_scenario):
      if q not in perfect_scenario_set and connection[q,i,j] == 0:
        m.addConstr(z[i,j,q] == 0,
          "bad_scenario[%s,%s,%s]" % (i, j, q) 
        )
    if not is_ip:
      for q in range(num_scenario-1):
        if q not in perfect_scenario_set:
          m.addConstr(z[i,j,q] <= 1,
              "z_upper_bound[%s,%s,%s]" % (i, j,q) 
          )

  for b in range(len(u_list)):
    u = u_list[b]
    v = v_list[b]
    w = w_list[b]
    q = scenario_list[b]
    m.addConstr(
          quicksum(cap[e1,e2] * u[e1,e2] * fe[q,e1,e2] for e1,e2 in arcs)
          + quicksum(v[i,j] for i,j in sd_pairs)
          + quicksum((z[i,j,q]-1) * w[i,j] for i,j in sd_pairs) <= alpha,
          "alpha_constr[%s,%s]" % (b, q) 
    )

  if step > 0:
    lhs, count_1 = 0, 0
    for q in range(num_scenario-1):
      if q not in perfect_scenario_set:
        for i, j in sd_pairs:
          if z_[i,j,q] == 1:
            count_1 += 1
            lhs = lhs - z[i,j,q]
          else:
            lhs = lhs + z[i,j,q]
    m.addConstr(
      count_1 + lhs <= step, "diff_bound" 
    )

  # Solve
  #m.Params.nodemethod = 2
  #m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 4
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)

  eps = 0.000000001

  store_z = {}
  for q in range(num_scenario):
    for i, j in sd_pairs:  
      if q not in perfect_scenario_set:
        store_z[i,j,q] = z[i,j,q].X
      else:
        store_z[i,j,q] = z_[i,j,q]
  changed_set = difference(store_z, z_, data)
  logging.info("number of changed scenarios: %s" % len(changed_set))

  total_diff = 0
  for q in range(num_scenario-1):
    if q not in perfect_scenario_set: 
      for i, j in sd_pairs:
        if z_[i,j,q] == 1:
          total_diff += z_[i,j,q] - z[i,j,q].X
        else:
          total_diff += z[i,j,q].X
  logging.info("total diff: %s" % total_diff)

  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return store_z, m.ObjVal, m.Runtime, changed_set
  return None, None

def smore_base_model(data, last_rhs, x):
  arcs = data['arcs']
  sd_pairs = data['sd_pairs']
  cap, demand = data['capacity'], data['demand']
  atunnels = data['atunnels']
  atraverse = data['atraverse']
  anum_tunnel = data['anum_tunnel']

  m = Model('smore')
  # Variable definition
  t = {}
  for l in range(anum_tunnel):
    x[l] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x[%s]" % (l))
  
  for i, j in sd_pairs:
    t[i,j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t[%s,%s]" % (i,j))

  theta = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")

  # Objective definition
  m.setObjective(theta, GRB.MINIMIZE)

  # Constraints definition
  for i, j in sd_pairs:
    m.addConstr(
      theta - t[i,j] >= 0,
      "theta[%s,%s]" % (i,j) 
    )
    last_rhs[i,j] = 0

  for e1, e2 in arcs:
    m.addConstr(
      quicksum(x[l] for l in atraverse[e1, e2]) <= cap[e1, e2],
        "capacity[%s,%s]" % (e1, e2)
    )
    last_rhs[e1, e2] = cap[e1, e2]

  for i, j in sd_pairs:
    m.addConstr(
      quicksum(x[l]/demand[i,j] for l in atunnels[i,j]) >= 1 - t[i,j],
      "demand[%s,%s]" % (i,j) 
    )  
  # Solve
  #m.Params.nodemethod = 2
  m.Params.method = 2
  #m.Params.Crossover = 0
  m.Params.Threads = 1
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)
  return m

def solve_smore(m, data, q, failed_edge_set, z_, last_rhs, x):
  # Parameters
  arcs = data['arcs']
  sd_pairs = data['sd_pairs']
  cap, demand = data['capacity'], data['demand']
  atunnels = data['atunnels']
  atraverse = data['atraverse']
  anum_tunnel = data['anum_tunnel']
  atunnel_edge_set = data['atunnel_edge_set']

  # Constraints definition
  for i, j in sd_pairs:
    if last_rhs[i,j] != z_[i,j,q] - 1:
      constr = m.getConstrByName("theta[%s,%s]" % (i,j))
      constr.setAttr("rhs", z_[i,j,q] - 1)
      last_rhs[i,j] = z_[i,j,q] - 1

  for e1, e2 in arcs:
    capacity = cap[e1, e2]
    if (e1, e2) in failed_edge_set:
      capacity = 0
    if last_rhs[e1, e2] != capacity:
      constr = m.getConstrByName("capacity[%s,%s]" % (e1, e2))
      constr.setAttr("rhs", capacity)
      last_rhs[e1, e2] = capacity

  # Solve
  #m.update()
  m.optimize()

  logging.info("Runtime: %f seconds" % m.Runtime)

  eps = 0.0000001
  store_x = {}
  for i in x:
    store_x[q,i] = x[i].X

  store_u = {}
  store_v = {}
  store_w = {}
  theta_n = {}
  capa_n = {}
  demand_n = {}
  for i, j in sd_pairs:
    name = "theta[%s,%s]" % (i,j) 
    theta_n[name] = (i,j)
    name = "demand[%s,%s]" % (i,j)
    demand_n[name] = (i,j)
  for e1, e2 in arcs:
    name = "capacity[%s,%s]" % (e1, e2)
    capa_n[name] = (e1, e2)
  constraints = m.getConstrs()
  check_dual = 0
  for constr in constraints:
    name = constr.constrName
    if name in theta_n:
      store_w[theta_n[name][0], theta_n[name][1]] = constr.pi
      check_dual += constr.pi * (z_[theta_n[name][0], theta_n[name][1], q] - 1)
    if name in capa_n:
      store_u[capa_n[name][0], capa_n[name][1]] = constr.pi
      if (capa_n[name][0], capa_n[name][1]) not in failed_edge_set:
        check_dual += constr.pi * cap[capa_n[name][0], capa_n[name][1]]  
    if name in demand_n:
      store_v[demand_n[name][0], demand_n[name][1]] = constr.pi
      check_dual += constr.pi
    
  if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
    return store_x, store_u, store_v, store_w, m.ObjVal, m.Runtime
  return None, None
