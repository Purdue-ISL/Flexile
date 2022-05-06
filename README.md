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

Currently supported schemes (updating): Teavar, Smore, Smore_connected, FlexileIP, FlexileBender, CvarFlowSt, CvarFlowAd, SwanThroughput, SwanMaxmin, FlexileIP2Class, FlexileBender2Class

1. Teavar: "TeaVaR: Striking the Right Utilization-Availability Balance in
WAN Traffic Engineering" SIGCOMM 2019
2. Smore: "Semi-Oblivious Traffic Engineering: The Road Not Taken" NSDI 2018
3. Smore_connected: Like Smore, but only considers connected flows in each scenario
4. FlexileIP, FlexileIP2Class: Model (I) in Flexile paper
5. FlexileBender, FlexileBender2Class: Refer to sec 3.2 in Flexile paper
6. CvarFlowSt, CvarFlowAd: Refer to sec 4 in Flexile paper
7. SwanThroughput, SwanMaxmin: "Achieving High Utilization with Software-Driven WAN" SIGCOMM 2013

## Configuration file and data file

### main configuration

The main configuration file specifies the scheme. For example in pcf/_config/main.yaml:

```bash
main:
    log_level: 'INFO'
    scheme: 'CvarFlowAd'  # Options: 'Teavar', 'Smore', 'Smore_connected', 'FlexileIP', 'FlexileBender', 'CvarFlowSt', 'CvarFlowAd', 'SwanThroughput', 'SwanMaxmin', 'FlexileBender2Class', 'FlexileIP2Class'
```

### topology configuration

The topology configuration file for single traffic class specifies 

1. beta: target availability
2. Total number of traffic matrices in the traffic matrix file
3. The index of traffic matrix to be used
4. File paths of capacity file, traffic matrix file, tunnel file and scenario file.
5. step: used only in FlexileBender or FlexileBender2Class model to specify the maximum number of critical flows allowed to change in all scenarios. Refer to sec 3.2 about 'Ensure better stability' in Flexile paper. 

For example in ./_config/toy_config.yaml:


```bash
name: 'toy'                   # Example in the paper
attributes:
    beta: 0.99                # Target Availbility
traffic_matrix:
    num_matrices: 1           # Number of traffic matrices in the traffic file
    tm_index: 0               # Traffic matrix index to be used
    step: 2000     # Used in FlexileBender or FlexileBender2Class model. 
data: 
    cap_file: '../_data/toy/toy_capa.tab'                  # Capacity file path
    tm_file: '../_data/toy/toy_traffic.tab'                # Traffic matrix file path
    tunnel_file: '../_data/toy/toy_tunnel.tab'             # Tunnel file path
    scenario_file: '../_data/toy/toy_scenarios.tab'        # Scenario file path
```

Note that even though in traffic matrix file, multiple matrices may be provided, only one traffic matrix will be used for one experiment. "tm_index" specifies that particular traffic matrix. See traffic matrix file for more details. 

The topology configuration file for two traffic class is similar to single traffic class with the following difference:

1. Additionally specifying the scale of low priority traffic
2. Two availability targets for two traffic classes
3. Two indexes for two traffic classes
4. 'h_tunnel': see 'tunnel file'
5. 'output_routing_file': see 'online routing'

For example in ./_config/ibm_2class_config.yaml:

```bash
name: 'ibm'
attributes:
    scale_low: 2	# Scale of low priority traffic
    beta_low: 0.999	# Target Availability for low priority traffic
    beta_high: 0.99	# Target Availability for high priority traffic
    step: 2000     # Used in FlexileBender or FlexileBender2Class model. Constrain the changes of critical scenarios from iteration to iteration.
traffic_matrix:
    num_matrices: 2	# Number of traffic matrices in the traffic file
    tm_index_low: 0	# Traffic matrix index to be used for low priority traffic
    tm_index_high: 1	# Traffic matrix index to be used for high priority traffic
data: 
    cap_file: '../_data/ibm/ibm_capa.tab'
    tm_file: '../_data/ibm/ibm_traffic_2class.tab'
    tunnel_file: '../_data/ibm/ibm_tunnel_2class.tab'
    scenario_file: '../_data/ibm/ibm_scenario.tab'
```

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

If the tunnel file is used for two traffic class, for each source-destination pair, all tunnels can be used to route low-priority traffic while by default, only the first 3 tunnels will be used for high-priority traffic. You can change this number by adding a field 'h_tunnel' under 'attributes' category in the topology configuration.

### scenario file

In the scenario file, each line specifies a failure scenario. A failure scenario consists of set of failed links and failure probability. For example in ./_data/toy/toy_scenarios.tab,

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

the first line specifies the normal case(no link fails) with 97.0299% probability and the second line says that the failure case where only 0-1 fails has probability of 0.9801%. Note that it is NOT necessary to list all failure scenarios. However, if the provided set of scenarios cannot cover the availability target, then the models cannot design for the target.

### result

The result file contains the logging from Gurobi and post analysis of the scheme.

#### percentile loss

After solving the model, the result file will print a 'profile' of percentile analysis. At the end, it will show the loss at the given percentile target (beta). See ./main/result.txt for reference.

#### online routing

If you are running 'FlexileBender2Class' scheme, a routing file will be generated representing the online routing achieved (refer to sec 3.3 in the Flexile paper). The format of this file is as below.

```bash
scenario tunnel priority allocation
0 0 l 0.5
0 1 l 0.5
0 0 h 0.5
0 1 h 0.5
```

By default, this file will be named as 'routing_file.txt'. You can change the file name and path by adding a field 'output_routing_file' under 'attributes' category in the topology configuration.

