import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util

from graph_ising import GraphIsing, TFGraph
from utils import timed, get_device

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
tf.autograph.set_verbosity(0, alsologtostdout=True)


def test_create():
    N = 10
    K = 10
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    # test exact sizes
    with timed('TFGraph'):
        tfg = TFGraph(g, N, N * 4)
    with timed('GraphIsing'):
        gis = GraphIsing(K, N, N * 4)
    with timed('set_graphs'):
        gis.set_graphs([tfg] * K)
    with timed('GraphIsing (with graphs)'):
        gis2 = GraphIsing([tfg] * K, N, N * 4)

    # smaller grapn can be set to gis
    with timed('set_graphs with smaller TFGraph'):
        g2 = nx.random_graphs.powerlaw_cluster_graph(N / 2, 3, 0.5)
        tfg2 = TFGraph(g2)
        gis.set_graphs([tfg2] * (2 * K // 3))
    assert (gis.v_node_masks.numpy()[0, :g2.order()] == True).all()
    assert (gis.v_node_masks.numpy()[0, g2.order():] == False).all()

    # gis with auto sizes
    with timed('auto sized TFGraph and GraphIsing'):
        g3 = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
        tfg3 = TFGraph(g3)
        gis3 = GraphIsing([tfg3] * K)
        gis.set_graphs([tfg2] * K)


def test_ops():
    graphs = [nx.path_graph(3), nx.path_graph(4), nx.path_graph(2), nx.Graph()]
    tfgs = [TFGraph(g) for g in graphs]
    gis = GraphIsing(tfgs, 4, 10)
    data = tf.constant([[2, 1, 3, 9], [1, 4, 2, -1], [3, 1, 9, 9], [6, 5, 0, -1]], dtype=tf.int64)
    assert (gis.max_neighbors(data).numpy() == [[1, 3, 1, 0], [4, 2, 4, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    assert (gis.sum_neighbors(data).numpy() == [[1, 5, 1, 0], [4, 3, 3, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    assert (gis.mean_neighbors(tf.cast(data, tf.float32)).numpy() == [[1, 2.5, 1, 0], [4, 1.5, 1.5, 2], [1, 3, 0, 0], [0, 0, 0, 0]]).all()
    edge_mask = tf.constant([1, 1, 0, 0] + [1] * 8)
    assert (gis.max_neighbors(data, edge_mask=edge_mask).numpy()[0] == [1, 2, 0, 0]).all()
    assert (gis.sum_neighbors(data, edge_mask=edge_mask).numpy()[0] == [1, 2, 0, 0]).all()


def test_update_and_caching():
    N = 1000
    K = 100
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    with timed('TFGraph and GraphIsing'):
        tfg = TFGraph(g)
        gis = GraphIsing([tfg] * K, N, N * 4)
        s0 = gis.initial_spins(-1.0)

    @tf.function
    def repeat(iters, data):
        for i in range(iters):
            data = gis.update(data, 0.5)
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
        return gis.largest_cluster(spins, max_iters=iters)

    graphs = [
        nx.Graph([(0,1), (0,2), (1,2), (2,3), (3,6), (5,4)]), # comp sizes 2, 5
        nx.Graph([(0,1), (0,2), (1,2), (3,6), (5,4)]), # comp sizes 2, 2, 3
        nx.Graph([(0,1), (0,2), (1,2), (2,5), (3,4), (5,4)]), # comp sizes 6, less vertices
    ]
    tfgs = [TFGraph(g) for g in graphs]
    gis = GraphIsing(tfgs, 10, 10)
    s0 = gis.initial_spins(1.0)

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
    tfg = TFGraph(g)
    gis = GraphIsing([tfg] * 4, 16, 30)
    s0 = tf.constant([
        [0, 1, 0, 1,  1, 1, 0, 1,  0, 1, 0, 1,  0, 1, 1, 0],
        [0, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 1],
        [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
        [1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1],
        ], tf.float32) * 2 - 1
    assert (gis.largest_cluster(s0).numpy() == [6, 7, 0, 16]).all()
 
    # test edge dropping
    assert (gis.largest_cluster(s0, drop_edges=0.0).numpy() == [6, 7, 0, 16]).all()
    assert (gis.largest_cluster(s0, drop_edges=1.0).numpy() == [1, 1, 0, 1]).all()

    # test averaged edge dropping
    smpl = gis.sampled_largest_cluster(s0, drop_edges=0.2, samples=tf.constant(50))
    assert (smpl.numpy() <= [6, 7, 0, 16]).all()
    assert (smpl.numpy() >= [3.5, 4, 0, 14.5]).all()


def test_bench():
    with tf.device("/device:GPU:0"):
        g = nx.grid_2d_graph(100, 100)
        tfg = TFGraph(g)
        gis = GraphIsing([tfg] * 100)
        s0 = gis.initial_spins()
        print("Graphs: 100 graphs, 100x100 grid (1M nodes)")

        @tf.function
        def repeat_update(spins, iters):
            for i in range(iters):
                spins = gis.update_op(spins, 0.5)
            return spins

        @tf.function
        def repeat_components(spins, iters):
            return gis.largest_cluster(spins, iters)

        @tf.function
        def repeat_sampled_components(spins, iters):
            return gis.sampled_largest_cluster(spins, iters)

        with timed('warmup'):
            repeat_update(s0, tf.constant(1))
        with timed('run 100x updates #1'):
            repeat_update(s0, tf.constant(100))
        with timed('run 100x updates #2'):
            repeat_update(s0, tf.constant(100))

        with timed('warmup'):
            repeat_components(s0, tf.constant(1))
        with timed('run 1x range 100 components #1'):
            repeat_components(s0, tf.constant(100))
        with timed('run 1x range 100 components #2'):
            repeat_components(s0, tf.constant(100))

        with timed('warmup'):
            repeat_sampled_components(s0, tf.constant(1))
        with timed('run 1x range 10 sampled components (10 samples) #1'):
            repeat_sampled_components(s0, tf.constant(10))
        with timed('run 1x range 10 sampled components (10 samples) #2'):
            repeat_sampled_components(s0, tf.constant(10))

