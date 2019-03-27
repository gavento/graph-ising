import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf

from graph_ising.forward_flux import DirectIsingClusterFFSampler
from graph_ising.utils import get_device, timed


tf.autograph.set_verbosity(0, alsologtostdout=True)

def main():
    K = 10  # Graphs
    N = 100  # Nodes per graph
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)

    ff = DirectIsingClusterFFSampler(g, [0, 5, 10, 15, 20], batch_size=K, T=2.0)
    with timed('warmup'):
        ff.run_batch_from(0, 0, 0.1)
    with timed('5000 steps, 1%% clusterings'):
        ff.run_batch_from(0, 5000, 0.2)

    print("0", ff.pops[0])
    print("1", ff.pops[1])


main()