import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf
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


def test_spin_components():
    g = nx.grid_2d_graph(4, 4)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
    gs = GraphSetWithConnectivity(graphs=[g] * 4)
    s0 = tf.reshape(tf.constant([
        [0, 1, 0, 1,  1, 1, 0, 1,  0, 1, 0, 1,  0, 1, 1, 0],
        [0, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 1],
        [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
        [1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1],
        ], tf.float32) * 2 - 1, [-1])
    node_mask = s0 > 0

    assert (gs.largest_components(node_mask=node_mask).numpy() == [6, 7, 0, 16]).all()
 
    # test edge dropping
    assert (gs.mean_largest_components(node_mask=node_mask, drop_edges=0.0, samples=1).numpy() == [6, 7, 0, 16]).all()
    assert (gs.mean_largest_components(node_mask=node_mask, drop_edges=1.0, samples=1).numpy() == [1, 1, 0, 1]).all()

    # test averaged edge dropping
    smpl = gs.mean_largest_components(node_mask=node_mask, drop_edges=0.2, samples=tf.constant(50))
    assert (smpl.numpy() <= [6, 7, 0, 16]).all()
    assert (smpl.numpy() >= [3.5, 4, 0, 14.5]).all()

    gs.print_metrics()
