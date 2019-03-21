import networkx as nx
import numpy as np
import tensorflow as tf
#from tensorflow.python.ops import control_flow_util

from graph_ising import GraphSet, ComponentsMixin
from graph_ising.utils import timed

#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
#tf.autograph.set_verbosity(0, alsologtostdout=True)


class GraphSetWithConnectivity(ComponentsMixin, GraphSet):
    pass


def test_graph_components():
    graphs = [
        nx.Graph([(0,1), (0,2), (1,2), (2,3), (3,6), (5,4)]), # comp sizes 2, 5
        nx.Graph([(0,1), (0,2), (1,2), (3,6), (5,4)]), # comp sizes 2, 2, 3
        nx.Graph([(0,1), (0,2), (1,2), (2,5), (3,4), (5,4)]), # comp sizes 6, less vertices
        ]

    gs = GraphSetWithConnectivity(graphs=graphs)

    @tf.function
    def comps(iters):
        return gs.largest_components(max_iters=iters)

    with timed('direct'):
        assert (gs.largest_components().numpy() == [5, 3, 6]).all()
    with timed('comps(3) #1'):
        assert (comps(tf.constant(3)).numpy() == [5, 3, 6]).all()
    with timed('comps(16) #1'):
        assert (comps(tf.constant(16)).numpy() == [5, 3, 6]).all()
    with timed('comps(2) #1'):
        assert (comps(tf.constant(2)).numpy() == [3, 3, 6]).all()
    with timed('comps(2) #2'):
        assert (comps(tf.constant(2)).numpy() == [3, 3, 6]).all()

    gs.print_metrics()

