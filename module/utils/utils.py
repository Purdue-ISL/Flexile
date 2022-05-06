"""
utils.py
Run with python2.7+ (but not python3)
"""

def output_routing(output_file, num_scenario, x):
  f = open(output_file, "w")
  f.write("scenario tunnel allocation\n")
  for q in range(num_scenario):
    for tunnel in x[q]:
      f.write("%d %d %f\n" % (q, tunnel, x[q][tunnel]))
  f.close()

def output_routing_2class(output_file, num_scenario, x_low, x_high):
  f = open(output_file, "w")
  f.write("scenario tunnel priority allocation\n")
  for q in range(num_scenario):
    for tunnel in x_low[q]:
      f.write("%d %d l %f\n" % (q, tunnel, x_low[q][tunnel]))
    for tunnel in x_high[q]:
      f.write("%d %d h %f\n" % (q, tunnel, x_high[q][tunnel]))
  f.close()
