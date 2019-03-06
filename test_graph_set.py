import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util

from graph_set import GraphIsing, TFGraph
from utils import timed

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
tf.autograph.set_verbosity(0, alsologtostdout=True)


def test_create():
    N = 1000
    K = 1000
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    with timed('TFGraph'):
        tfg = TFGraph(g, N, N * 4)
    with timed('GraphIsing'):
        gis = GraphIsing(K, N, N * 4)
    with timed('set_graphs'):
        gis.set_graphs([tfg] * K)
    with timed('GraphIsing (with graphs)'):
        gis2 = GraphIsing([tfg] * K, N, N * 4)


def test_ops():
    graphs = [nx.path_graph(3), nx.path_graph(4), nx.path_graph(2), nx.Graph()]
    tfgs = [TFGraph(g, 4, 10) for g in graphs]
    gis = GraphIsing(tfgs, 4, 10)
    data = tf.constant([[2, 1, 3, 9], [1, 4, 2, -1], [3, 1, 9, 9], [6, 5, 0, -1]], dtype=tf.int32)
    assert (gis.max_neighbors_op(data).numpy() == [[1, 3, 1, 0], [4, 2, 4, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    assert (gis.sum_neighbors_op(data).numpy() == [[1, 5, 1, 0], [4, 3, 3, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    assert (gis.mean_neighbors_op(tf.cast(data, tf.float32)).numpy() == [[1, 2.5, 1, 0], [4, 1.5, 1.5, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    edge_mask = tf.constant([1, 1, 0, 0] + [1] * 8)
    assert (gis.max_neighbors_op(data, edge_mask=edge_mask).numpy()[0] == [1, 2, 0, 0]).all()
    assert (gis.sum_neighbors_op(data, edge_mask=edge_mask).numpy()[0] == [1, 2, 0, 0]).all()


def test_update_and_caching():
    N = 1000
    K = 100
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    with timed('TFGraph and GraphIsing'):
        tfg = TFGraph(g, N, N * 4)
        gis = GraphIsing([tfg] * K, N, N * 4)
        s0 = gis.initial_spins(-1.0)

    @tf.function
    def repeat(iters, data):
        for i in range(iters):
            data = gis.update_op(data, 0.5)
        return data

    with timed('single #1'):
        s = repeat(1, s0)
    with timed('single #2'):
        s = repeat(1, s0)
    with timed('repeated #1'):
        s = repeat(3, s0)
    with timed('repeated #2'):
        s = repeat(3, s0)
    #print([cf.structured_input_signature for cf in repeat._list_all_concrete_functions_for_serialization()])
    assert len(repeat._list_all_concrete_functions_for_serialization()) == 2

    gis.set_graphs([tfg] * K)
    with timed('single #3'):
        s = repeat(1, s0)
    assert len(repeat._list_all_concrete_functions_for_serialization()) == 2

def test_comps():
    @tf.function
    def comps(gset, iters):
        return gset.components_op(iters)

    graphs = [
        nx.Graph([(0,1), (0,2), (1,2), (2,3), (3,6), (5,4)]), # comp sizes 2, 5
        nx.Graph([(0,1), (0,2), (1,2), (3,6), (5,4)]), # comp sizes 2, 2, 3
        nx.Graph([(0,1), (0,2), (1,2), (2,5), (3,4), (5,4)]), # comp sizes 6, less vertices
    ]
    gs = GraphSet(graphs)
    gs.construct()

    with timed('gs'):
        assert (gs.components_op().numpy() == [5, 3, 6]).all()
    with timed('gs'):
        assert (comps(gs, 3).numpy() == [5, 3, 6]).all()
    with timed('gs'):
        assert (comps(gs, 2).numpy() == [3, 3, 6]).all()
    with timed('gs rep'):
        assert (comps(gs, 2).numpy() == [3, 3, 6]).all()

    gs2 = GraphSet([graphs[1], graphs[0], nx.path_graph(8)])
    gs2.construct()
    with timed('gs2'):
        assert (gs2.components_op().numpy() == [3, 5, 8]).all()
    with timed('gs2'):
        assert (comps(gs2, tf.constant(3)).numpy() == [3, 5, 4]).all()
    with timed('gs2'):
        assert (comps(gs2, tf.constant(2)).numpy() == [3, 3, 3]).all()
    with timed('gs2 rep'):
        assert (comps(gs2, tf.constant(2)).numpy() == [3, 3, 3]).all()

    for f in [comps, gs.components_op, gs.components_op, GraphSet.components_op]:
        with timed('sigs'):
            print([(l, l.structured_input_signature) for l in f._list_all_concrete_functions_for_serialization()])
