import networkx as nx
import numpy as np
import tensorflow as tf
#from tensorflow.python.ops import control_flow_util

from graph_ising import GraphSet
from graph_ising.utils import timed

#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
#tf.autograph.set_verbosity(0, alsologtostdout=True)


def test_create():
    N = 200
    K = 200
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    # test exact sizes
    with timed('GraphSet'):
        tfg = GraphSet(graphs=[g] * K)


def test_ops():
    graphs = [nx.path_graph(3), nx.path_graph(4), nx.path_graph(2), nx.Graph()]
    gs = GraphSet(graphs=graphs)
    data = tf.constant([2, 1, 3, 9, 1, 4, 2, -1, 3], dtype=tf.int64)
    assert (gs.max_neighbors(data).numpy() == [1, 3, 1, 1, 9, 2, 4, 3, -1]).all()
    assert (gs.sum_neighbors(data).numpy() == [1, 5, 1, 1, 13, 3, 4, 3, -1]).all()
    assert (gs.mean_neighbors(tf.cast(data, tf.float32)).numpy() == [1, 2.5, 1, 1, 6.5, 1.5, 4, 3, -1]).all()
    edge_mask = tf.constant([1, 1, 0, 0] + [1] * 8)
    assert (gs.max_neighbors(data, edge_weights=edge_mask).numpy()[:3] == [1, 2, 0]).all()
    assert (gs.sum_neighbors(data, edge_weights=edge_mask).numpy()[:3] == [1, 2, 0]).all()

