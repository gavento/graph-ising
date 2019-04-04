import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf

from graph_ising.forward_flux import DirectIsingClusterFFSampler
from graph_ising.utils import get_device, timed


tf.autograph.set_verbosity(0, alsologtostdout=True)

def main():
    K = 10  # Graphs
    #N = 100  # Nodes per graph
    #g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    g = nx.grid_2d_graph(10, 10)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')

    ff = DirectIsingClusterFFSampler(g, [0, 5, 10, 15, 20], batch_size=K, T=1.5)
    with timed('warmup'):
        ff.run_batch_from(0, 0, 0.1)
    with timed('run 1'):
        ff.run_batch_from(0, 1000, 0.3)
    with timed('run 2'):
        ff.run_batch_from(0, 2000, 0.3)
    with timed('run 3'):
        ff.run_batch_from(1, 2000, 0.3)
    for i, v in enumerate(ff.interfaces):
        print("Interface", v)
        for p in ff.pops[i]:
            print("  ", p)


main()