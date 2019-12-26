import pytest
import networkx as nx
from netising import utils, cising


def test_bench():
    N = 100
    K = 100
    print("Grid {}x{}".format(N, N))
    g = nx.grid_2d_graph(N, N)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
    s = cising.IsingState(graph=g, T=1.1, F=0.1)

    with utils.timed("Clustering {} vertices".format(g.order()), iters=K):
        for i in range(K):
            cs = s.mc_max_cluster(value=-1)
            assert cs.v_in == N * N

    with utils.timed("Updating {} vertices".format(g.order()), iters=K):
        s.mc_sweep(K)


def test_base():

    n = 1000
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    print(g.size(), g.order())

    s = cising.IsingState(graph=g, F=1.5, T=8.0)
    assert s.get_hamiltonian() == 1.5 * n - g.size()

    s1 = s.copy()
    s1.mc_random_update(10000)
    r1 = s1.mc_max_cluster(samples=3, edge_prob=0.9)

    s2 = s.copy()
    s2.mc_random_update(10000)
    r2 = s2.mc_max_cluster(samples=3, edge_prob=0.9)

    assert s1 == s2
    assert s1 != s
    assert r1 == r2

    s3 = s.copy()
    s3.mc_sweep(50)
    s3.mc_sweep(50)
    r3 = s3.mc_max_cluster(samples=1, edge_prob=1.0)
    
    s4 = s.copy()
    s4.mc_sweep(100)
    r4 = s4.mc_max_cluster(samples=1, edge_prob=1.0)

    assert s3 == s4
    assert s3 != s
    assert r3 == r4
