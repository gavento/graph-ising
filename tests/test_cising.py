import networkx as nx
import pytest

from netising import utils
from netising.ising_state import GraphIsingState, SpinCountIsingState


def test_bench():
    N = 100
    K = 100
    print("Grid {}x{}".format(N, N))
    g = nx.grid_2d_graph(N, N)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
    s = SpinCountIsingState(graph=g, T=1.1, F=0.1)

    with utils.timed("Updating {} vertices (batch=1000)".format(g.order()), iters=K):
        s.mc_updates(1000)

    with utils.timed("Updating {} vertices (batch=1)".format(g.order()), iters=K):
        s.mc_updates(1)

def test_base():

    n = 1000
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)

    s = GraphIsingState(graph=g, F=1.5, T=8.0)
    assert s.get_hamiltonian() == 1.5 * n - g.size()

    s1 = s.copy()
    s2 = s.copy()
    s3 = s.copy()

    assert s1.seed == s2.seed

    s1.mc_updates(1000)
    s2.mc_updates(1000)
    assert s1.seed == s2.seed
    assert s.seed != s2.seed

    s3.mc_updates(500)
    s3.mc_updates(500)

    assert s1 != s
    assert tuple(s1.spins) == tuple(s2.spins)
    assert tuple(s1.spins) == tuple(s3.spins)

