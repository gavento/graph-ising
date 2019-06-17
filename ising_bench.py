import concurrent.futures
import ising
import networkx as nx
import numpy as np
import time
import timeit


def bench():
    N = 100
    K = 1000
    print("Grid {}x{}, {} times".format(N, N, K))
    g = nx.grid_2d_graph(N, N)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
    s = ising.IsingState(graph=g, T=1.1, field=0.1)

    t0 = time.perf_counter()
    for i in range(K):
        cs = s.mc_max_cluster(value=1)
        assert cs.v_in == N * N
    t = time.perf_counter() - t0
    print("{} node clusterings in {} s: {} / s".format(K * N * N, t, (K * N * N) / t))

    t0 = time.perf_counter()
    s.mc_sweep(K)
    t = time.perf_counter() - t0
    print("{} node updates in {} s: {} / s".format(K * N * N, t, (K * N * N) / t))


bench()
