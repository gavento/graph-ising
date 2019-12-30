import networkx as nx
import numpy as np

from netising.ising_graph import IsingGraph


def test_basic():
    g = nx.Graph([[0, 1], [1, 2], [0, 3], [0, 2]])
    ig = IsingGraph(g)
    cig = ig.get_ising_graph()
    assert cig.n == 4
    assert [cig.degree[i] for i in range(cig.n)] == [3, 2, 2, 1]
    #assert cig.degree_sum == [3, 5, 7, 8]
    assert ([cig.neigh_cap[i] for i in range(cig.n)] >= np.array([3, 2, 2, 1])).all()
    for v in g.nodes():
        assert set(cig.neigh_list[v][i] for i in range(g.degree(v))) == set(g.neighbors(v))
