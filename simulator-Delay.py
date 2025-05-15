"""
End-to-end toy simulator
────────────────────────
1.  Build a 18 000-node Barabási–Albert topology.
2.  Split the network into three logical layers (MH / ML / IM).
3.  Attach an M/M/1 queue to every node with **SimPy**:
      • Poisson arrivals (λ)              →  random inter-arrivals  
      • Exponential service               →  random service time
4.  Fire N read requests per layer and measure
      (network latency  +  queueing delay  +  service time).
5.  Show an “ECG-style” smoothed plot with the
   evolution of delays by class.
"""
import random
import simpy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ------------------------------------------------------------------
# ▌1▐  GLOBAL PARAMETERS
# ------------------------------------------------------------------
TOTAL_NODES      = 18_000              # graph size
LAYER_SIZE       = 6_000               # nodes per logical layer
REPL_IM          = 13                  # IM replicas  (≈ log₂ 6000)
SAMPLES_PER_CLS  = 100_000             # requests to simulate per class
DELAY_HOP_MS     = 120                 # fixed latency per hop
#
TX_MEAN_MS       = 10                  # μ service   (transactions)
BLK_MEAN_MS      = 50                  # μ service   (blocks)
QUEUE_CAPACITY   = 1                   # M/M/1  (change → M/M/c)
INTARRIVAL_MS    = 5                   # λ arrivals (mean 5 ms)
#
rng = random.Random(42)                # reproducible seed

# ------------------------------------------------------------------
# ▌2▐  BA GRAPH  +  LAYERS  +  REPLICAS
# ------------------------------------------------------------------
G = nx.barabasi_albert_graph(TOTAL_NODES, m=8, seed=42)
if not nx.is_connected(G):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

nodes = list(G.nodes()); rng.shuffle(nodes)

mh_nodes = nodes[:LAYER_SIZE]
ml_nodes = nodes[LAYER_SIZE:2*LAYER_SIZE]
im_nodes = nodes[2*LAYER_SIZE:]
im_repl  = rng.sample(im_nodes, REPL_IM)

layer_sets = {"ML": ml_nodes, "MH": mh_nodes, "IM": im_repl}
colors     = {"MH": "blue", "ML": "gold", "IM": "red"}

# ------------------------------------------------------------------
# ▌3▐  SIMPY MODEL PER NODE
# ------------------------------------------------------------------
class SimNode:
    """Node with two independent M/M/1 queues (tx and block)."""

    def __init__(self, env: simpy.Environment):
        self.tx_q  = simpy.Resource(env, capacity=QUEUE_CAPACITY)
        self.blk_q = simpy.Resource(env, capacity=QUEUE_CAPACITY)
        self.env   = env

    # --- exponential services ------------------------------------
    def serve_tx(self):
        dur = rng.expovariate(1 / TX_MEAN_MS)   # Markovian tx service
        yield self.env.timeout(dur)

    def serve_blk(self):
        dur = rng.expovariate(1 / BLK_MEAN_MS)  # Markovian block service
        yield self.env.timeout(dur)

# Dictionary of simulated nodes
env        = simpy.Environment()
nodes_sim  = {n: SimNode(env) for n in G.nodes()}

# ------------------------------------------------------------------
# ▌4▐  AUXILIARY FUNCTIONS
# ------------------------------------------------------------------
def min_hops(src, targets):
    """Shortest-path hop count; 1 if local read."""
    if src in targets:
        return 1
    lengths = nx.single_source_shortest_path_length(G, src)
    dists   = [lengths[t] for t in targets if t in lengths]
    return min(dists) if dists else None        # None → no path

def single_query(env, cls_label, stats):
    """
    One request:
    – Pick source + destination
    – Travel through the network (fixed hop latency)
    – Queue and serve at destination
    – Record total delay
    """
    src     = rng.choice(nodes)
    targets = layer_sets[cls_label]
    hops    = min_hops(src, targets)
    if hops is None:
        return                                   # isolated node (very rare)

    # 1) network delay
    net_ms = hops * DELAY_HOP_MS

    # 2) select destination (IM → any of the 13 replicas)
    dst     = targets[0] if cls_label != "IM" else rng.choice(targets)
    node    = nodes_sim[dst]

    # 3) acquire the proper resource
    queue   = node.tx_q if cls_label != "IM" else node.blk_q
    with queue.request() as req:
        yield req
        start = env.now
        if cls_label != "IM":
            yield env.process(node.serve_tx())
        else:
            yield env.process(node.serve_blk())
        proc_ms = env.now - start                # queue + service time

    stats.append(net_ms + proc_ms)               # store total delay

def generator(env, cls_label, out_list):
    """Generate SAMPLES_PER_CLS requests with Poisson arrivals."""
    for _ in range(SAMPLES_PER_CLS):
        env.process(single_query(env, cls_label, out_list))
        ia = rng.expovariate(1 / INTARRIVAL_MS)  # next inter-arrival
        yield env.timeout(ia)

# ------------------------------------------------------------------
# ▌5▐  RUN SIMULATION
# ------------------------------------------------------------------
stats = {"ML": [], "MH": [], "IM": []}
for lbl in stats:
    env.process(generator(env, lbl, stats[lbl]))

env.run()       # run until all processes finish

# ------------------------------------------------------------------
# ▌6▐  PLOT RESULTS (simple smoothing)
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for lbl, data in stats.items():
    series = np.array(data)
    series = series[:100_000]
    smooth = np.convolve(series, np.ones(30)/30, mode='valid')

    # ← added: moving standard deviation
    stds = [np.std(series[i:i+30]) for i in range(len(series) - 30 + 1)]
    stds = np.array(stds)

    x_raw  = np.arange(len(smooth))
    spline = make_interp_spline(x_raw, smooth, k=3)
    x_s    = np.linspace(x_raw.min(), x_raw.max(), 500)
    y_s    = spline(x_s)

    # ← added: error band
    spline_std = make_interp_spline(x_raw, stds, k=3)
    y_err = spline_std(x_s)
    plt.fill_between(x_s, y_s - y_err, y_s + y_err,
                     color=colors[lbl], alpha=0.2)

    plt.plot(x_s, y_s, label=lbl, color=colors[lbl])

# availability threshold reference line
plt.axhline(500, color='gray', linestyle='--', linewidth=1.2,
            label="Availability threshold (500 ms)")

plt.title("Access Availability under SimPy Simulation\n(18,000 Nodes, M/M/1 Queues)")
plt.xlabel("Request index")
plt.ylabel("Estimated Delay (ms)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("simpy_network_total_delay.png", dpi=300)
# plt.show()
