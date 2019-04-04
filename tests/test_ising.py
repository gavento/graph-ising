import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf
#from tensorflow.python.ops import control_flow_util

from graph_ising import GraphSetIsing
from graph_ising.utils import get_device, timed


#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
#tf.autograph.set_verbosity(0, alsologtostdout=True)

def test_create():
    N = 10  # Graphs
    V = 10  # Nodes per graph
    g = nx.random_graphs.powerlaw_cluster_graph(V, 3, 0.5)
    gsi = GraphSetIsing(graphs=[g] * N)


def test_update():
    N = 2  # Graphs
    V = 10  # Nodes per graph
    g = nx.random_graphs.powerlaw_cluster_graph(V, 3, 0.5)
    gsi = GraphSetIsing(graphs=[g] * N, T=10.0)

    gsi.update([-1.0] * gsi.order)

    @tf.function
    def f(vec):
        gsi.update(vec)
    f(tf.constant([-1.0] * gsi.order))


def test_spin_components():
    g = nx.grid_2d_graph(4, 4)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
    gs = GraphSetIsing(graphs=[g] * 4)
    s0 = tf.reshape(tf.constant([
        [0, 1, 0, 1,  1, 1, 0, 1,  0, 1, 0, 1,  0, 1, 1, 0],
        [0, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 1],
        [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
        [1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1],
        ], tf.float32) * 2 - 1, [-1])

    assert (gs.largest_clusters(s0).numpy() == [6, 7, 0, 16]).all()
    smpl = gs.largest_clusters(s0, drop_edges=tf.constant(0.2), samples=tf.constant(50))
    assert (smpl.numpy() <= [6, 7, 0, 16]).all()
    assert (smpl.numpy() >= [3.5, 4, 0, 14.5]).all()

    gs.print_metrics()