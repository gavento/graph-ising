import contextlib
import time

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util

from graph_set import GraphSet

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


@contextlib.contextmanager
def timed(name=None):
    t0 = time.time()
    yield
    t1 = time.time()
    print((name + " " if name else "") + "took {:.3f} ms".format((t1 - t0) * 1000.0))


def test_ops():
    graphs = [nx.path_graph(3), nx.path_graph(4), nx.path_graph(2)]
    gs = GraphSet(graphs)
    gs.construct()
    data = np.array([[2, 1, 3, 9], [1, 4, 2, -1], [3, 1, 9, 9]])
    assert (gs.max_neighbors_op(data).numpy() == [[1, 3, 1, 0], [4, 2, 4, 2], [1, 3, 0, 0]]).all()
    assert (gs.sum_neighbors_op(data).numpy() == [[1, 5, 1, 0], [4, 3, 3, 2], [1, 3, 0, 0]]).all()
    assert (gs.mean_neighbors_op(data.astype(float)).numpy() == [[1, 2.5, 1, 0], [4, 1.5, 1.5, 2], [1, 3, 0, 0]]).all()


def test_comps():
    graphs = [
        nx.Graph([(0,1), (0,2), (1,2), (2,3), (3,6), (5,4)]), # comp sizes 2, 5
        nx.Graph([(0,1), (0,2), (1,2), (3,6), (5,4)]), # comp sizes 2, 2, 3
        nx.Graph([(0,1), (0,2), (1,2), (2,5), (3,4), (5,4)]), # comp sizes 6, less vertices
    ]
    gs = GraphSet(graphs)
    gs.construct()
    print(gs.components_op())


def test_update():
    graphs = [nx.path_graph(100)] * 100
    with timed('init graph set'):
        gs = GraphSet(graphs)
        gs.construct()

    @tf.function
    def single(s):
        return gs.update_op(s, 0.5, False)

    with timed('single #1'):
        s = single(gs.v_spins)
    with timed('single #2'):
        s = single(gs.v_spins)

    print(tf.autograph.to_code(gs.update_op, experimental_optional_features=None))

    @tf.function
    def repeated(s):
        for i in range(10):
            s = gs.update_op(s, 0.5, True)
        return s

    with timed('repeated #1'):
        s = repeated(gs.v_spins)
    with timed('repeated #2'):
        s = repeated(gs.v_spins)
