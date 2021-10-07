# Flexile

Design routing schemes to minimize loss for a certain availability.

Required library: [gurobi](https://www.gurobi.com/), yaml. Please add local gurobi path in ./main/run.py if needed.

## How to use:

Execute ./main/run.py with a main configuration file and a topology configuration file. For example,

```bash
cd ./main
python run.py --main_config ../_config/main.yaml --topo_config ../_config/toy_config.yaml
```

The main configuration file specifies the scheme. The topology configuration file specifies the topology information including the capacity file, the traffic file and the tunnel file. See ./_config/main.yaml and ./_config/b4_config.yaml for more detailed format.

Currently supported schemes (updating): Teavar, Smore, Smore_connected, FlexileIP, FlexileBender

## Configuration file and data file

### main configuration

The main configuration file specifies the scheme. For example in pcf/_config/main.yaml:

```bash
main:
    log_level: 'INFO'
    scheme: 'Teavar'  # Options: 'Teavar', 'Smore', 'Smore_connected', 'FlexileIP', 'FlexileBender'
```

### topology configuration

The topology configuration file specifies 

1. Beta: target availability
2. Total number of traffic matrices in the traffic matrix file
3. The index of traffic matrix to be used
4. File paths of capacity file, traffic matrix file, tunnel file and scenario file.

For example in ./_config/toy_config.yaml:


```bash
name: 'toy'                   # Example in the paper
attributes:
    beta: 0.99                # Target Availbility
traffic_matrix:
    num_matrices: 1           # Number of traffic matrices in the traffic file
    tm_index: 0               # Traffic matrix index to be used
data: 
    cap_file: '../_data/toy/toy_capa.tab'                  # Capacity file path
    tm_file: '../_data/toy/toy_traffic.tab'                # Traffic matrix file path
    tunnel_file: '../_data/toy/toy_tunnel.tab'             # Tunnel file path
    scenario_file: '../_data/toy/toy_scenarios.tab'        # Scenario file path
```

Note that even though in traffic matrix file, multiple matrices may be provided, only one traffic matrix will be used for one experiment. "tm_index" specifies that particular traffic matrix. See traffic matrix file for more details. 

### capacity file

In the capacity file, each line(in the format of 'i j c') specifies a direct link with capacity of c from i to j. For example in ./_data/toy/toy_capa.tab,

```bash
i j cap
0 1 1
1 0 1
1 2 1
2 1 1
0 2 1
2 0 1
```

This file specifies 6 links 0->1, 1->0, 1->2, 2->1, 0->2, 2->0 all with capacity of 1 unit. Note that the unit of capacity is not specified since only the relative values matter. And make sure that the unit of capacity is aligned with the unit of traffic. 

### traffic matrix file

We allow users to provide 1 or more traffic matrices in the traffic matrix file. Different traffic matrices are distinguished by indexes. Each line(in the format of 's t h tm') specifies that for the traffic matrix with index h, the demand from s to t is tm. For example in ./_data/toy/toy_traffic.tab,

```bash
s t h tm
0 1 0 1
0 2 0 1
```

the first 3 lines specify the demand from 0->1, 0->2 in the traffic matrix 0. Note that, only one traffic matrix will be used for one experiment according to the provided index in the configuration file. 
 
### tunnel file

In the tunnel file, each line specifies a physical tunnel. A physical tunnel has 4 entries in a line(source, destination, index, list of edges). For example in ./_data/toy/toy_tunnel.tab,

```bash
s t k edges
0 1 0 0-1
0 1 1 0-2,2-1
0 2 0 0-2
0 2 1 0-1,1-2
```

there are 2 physical tunnels from 0 to 1: [0-1] and [0-2,2-1], and 2 physical tunnels from 0 to 2: [0-2] and [0-1,1-2].

### scenario file

In the scenario file, each line specifies a failure scnario. A failure scenario consists of set of failed links and failure probability. For example in ./_data/toy/toy_scenarios.tab,

```bash
no 0.970299
0-1,1-0 0.009801
0-2,2-0 0.009801
1-2,2-1 0.009801
0-1,1-0,0-2,2-0 0.000099
0-1,1-0,1-2,2-1 0.000099
0-2,2-0,1-2,2-1 0.000099	
0-1,1-0,0-2,2-0,1-2,2-1 0.000001
```

the first line specifies the normal case(no link fails) with 97.0299% probability and the second line says that the failure case where only 0-1 fails has probability of 0.9801%. Note that it is NOT necessary to list all failure scenarios.
