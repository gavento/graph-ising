import concurrent.futures
import time

import networkx as nx
import numpy as np

from netising.ising_state import GraphIsingState


def _test_basic():
    g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    # Test 0
    s = GraphIsingState(g, spins=[-1, -1, -1, 1, 1])
    stat = s.mc_max_cluster(value=-1)
    assert stat.v_in == 3
    assert stat.v_in_border == 2
    assert stat.v_out_border == 2
    assert stat.e_in == 2
    assert stat.e_border == 2
