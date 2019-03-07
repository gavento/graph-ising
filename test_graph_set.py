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
        s = repeat(tf.constant(1), s0)
    with timed('single #2'):
        s = repeat(tf.constant(1), s0)
    with timed('repeated(10) #1'):
        s = repeat(tf.constant(10), s0)
    with timed('repeated(10) #2'):
        s = repeat(tf.constant(10), s0)
    #print([cf.structured_input_signature for cf in repeat._list_all_concrete_functions_for_serialization()])
    assert len(repeat._list_all_concrete_functions_for_serialization()) == 1

    gis.set_graphs([tfg] * K)
    with timed('single #3'):
        s = repeat(tf.constant(1), -s0)
    assert len(repeat._list_all_concrete_functions_for_serialization()) == 1


def test_graph_components():
    @tf.function
    def comps(spins, iters):
        return gis.largest_component_size_op(spins, iters=iters)

    graphs = [
        nx.Graph([(0,1), (0,2), (1,2), (2,3), (3,6), (5,4)]), # comp sizes 2, 5
        nx.Graph([(0,1), (0,2), (1,2), (3,6), (5,4)]), # comp sizes 2, 2, 3
        nx.Graph([(0,1), (0,2), (1,2), (2,5), (3,4), (5,4)]), # comp sizes 6, less vertices
    ]
    tfgs = [TFGraph(g, 10, 10) for g in graphs]
    gis = GraphIsing(tfgs, 10, 10)
    s0 = gis.initial_spins(1.0)

    with timed('direct'):
        assert (gis.largest_component_size_op(s0).numpy() == [5, 3, 6]).all()
    with timed('comps(3) #1'):
        assert (comps(s0, tf.constant(3)).numpy() == [5, 3, 6]).all()
    with timed('comps(16) #1'):
        assert (comps(s0, tf.constant(16)).numpy() == [5, 3, 6]).all()
    with timed('comps(2) #1'):
        assert (comps(s0, tf.constant(2)).numpy() == [3, 3, 6]).all()
    with timed('comps(2) #2'):
        assert (comps(s0, tf.constant(2)).numpy() == [3, 3, 6]).all()


def test_spin_components():
    g = nx.grid_2d_graph(4, 4)
    tfg = TFGraph(g, 16, 30)
    gis = GraphIsing([tfg] * 4, 16, 30)
    s0 = tf.constant([
        [0, 1, 0, 1,  1, 1, 0, 1,  0, 1, 0, 1,  0, 1, 1, 0],
        [0, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 1],
        [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
        [1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1],
        ], tf.float32) * 2 - 1
    assert (gis.largest_component_size_op(s0).numpy() == [6, 7, 0, 16]).all()
 
    # test edge dropping
    assert (gis.largest_component_size_op(s0, drop_edges=0.0).numpy() == [6, 7, 0, 16]).all()
    assert (gis.largest_component_size_op(s0, drop_edges=1.0).numpy() == [1, 1, 0, 1]).all()

    # test averaged edge dropping
    smpl = gis.sampled_largest_component_size_op(s0, drop_edges=0.2, drop_samples=tf.constant(50))
    assert (smpl.numpy() <= [6, 7, 0, 16]).all()
    assert (smpl.numpy() >= [3.5, 4, 0, 14.5]).all()
