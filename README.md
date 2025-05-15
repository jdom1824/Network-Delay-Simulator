# End-to-End Network Delay Simulator

This is an end-to-end network simulator that models the behavior of a large-scale topology using M/M/1 queues to evaluate the performance of different types of requests.

## Overview

The simulator builds an 18,000-node network using the **Barabási–Albert** model, representing a distributed infrastructure divided into three logical layers:

- MH (Max Hierarchy)
- ML (Mid Hierarchy)
- IM (Infrastructure Minimum)

Each node includes **M/M/1** queues that process requests asynchronously.

## Simulation Steps

1. **Network Construction**  
   An 18,000-node graph is generated using the Barabási–Albert model, ensuring full connectivity.

2. **Logical Layer Assignment**  
   Nodes are split evenly into three logical layers (6,000 nodes each), and 13 replicas are selected for the IM layer.

3. **Service Model**  
   Each node has two M/M/1 queues:
   - One for transactions (`TX`)
   - One for blocks (`BLK`)
   
   Arrival processes follow a **Poisson distribution**, and service times follow an **exponential distribution**.

4. **Request Simulation**  
   For each class (MH, ML, IM), 100,000 requests are generated. Each request travels through the network, experiences queuing and service delay, and the total response time is recorded as:  
   `network delay + queuing delay + service time`

5. **Result Visualization**  
   The script generates a smooth ECG-style plot showing delays per class over time, along with ±1 standard deviation error bands and a reference line at 500 ms to represent an availability threshold.

## Key Parameters

| Parameter             | Value                  | Description                                      |
|-----------------------|------------------------|--------------------------------------------------|
| `TOTAL_NODES`         | 18,000                 | Size of the graph                                |
| `LAYER_SIZE`          | 6,000                  | Nodes per logical layer                          |
| `REPL_IM`             | 13                     | IM layer replicas                                |
| `SAMPLES_PER_CLS`     | 100,000                | Requests per class                               |
| `DELAY_HOP_MS`        | 120 ms                 | Fixed latency per network hop                    |
| `TX_MEAN_MS`          | 10 ms                  | Mean service time for transactions               |
| `BLK_MEAN_MS`         | 50 ms                  | Mean service time for blocks                     |
| `QUEUE_CAPACITY`      | 1                      | Queue capacity (M/M/1 model)                     |
| `INTARRIVAL_MS`       | 5 ms                   | Mean inter-arrival time (Poisson process)        |

## Output

The script generates an image file:


The plot shows smoothed delay curves for each class (`MH`, `ML`, `IM`), with error bands (±1 standard deviation) and a dashed line at 500 ms indicating the availability threshold.

## Requirements

pip install simpy networkx matplotlib numpy scipy

