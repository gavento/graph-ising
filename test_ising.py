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
