import networkx as nx
import numpy as np
import pytest
import tensorflow.compat.v2 as tf

from graph_ising import GraphSet, GraphSetIsing
from graph_ising.utils import timed

#from tensorflow.python.ops import control_flow_util


@pytest.mark.slow
def test_bench():
    with tf.device("/device:CPU:0"):
        N = 1000
        K = 100
        g = nx.random_graphs.powerlaw_cluster_graph(N, 5, 0.5)
        with timed('construct graph set'):
            gsi = GraphSetIsing(graphs=[g] * K, T=5.0)
        s0 = gsi.initial_spins(-1.0)
        s1 = gsi.initial_spins(1.0)
        print("Graphs: {} powerlaw graphs, {} nodes ({} tot nodes)".format(K, N, N * K))

        @tf.function
        def repeat_update(spins, iters):
            iters = tf.identity(iters)
            for i in range(iters):
                spins = gsi.update(spins, 0.5)
            return spins

        @tf.function
        def repeat_clusters(spins, iters):
            iters = tf.identity(iters)
            for i in range(iters):
                gsi.largest_clusters(spins)

        @tf.function
        def repeat_mean_clusters(spins, iters):
            iters = tf.identity(iters)
            for i in range(iters):
                gsi.largest_clusters(spins, drop_edges=0.1, samples=10)

        with timed('update warmup'):
            repeat_update(s0, tf.constant(1))
        with timed('run 100x updates #1'):
            repeat_update(s0, tf.constant(100))
        with timed('run 100x updates #2'):
            repeat_update(s0, tf.constant(100))
        with timed('run 100x eager updates'):
            for i in range(100):
                gsi.update(s0, 0.5)

        with timed('components warmup'):
            repeat_clusters(s1, tf.constant(1))
        with timed('run 10x components #1'):
            repeat_clusters(s1, tf.constant(10))
        with timed('run 10x components #2'):
            repeat_clusters(s1, tf.constant(10))
        with timed('run 10x eager components'):
            for i in range(10):
                gsi.largest_clusters(s1)

        with timed('mean_clusters warmup'):
            repeat_mean_clusters(s1, tf.constant(1))
        with timed('run 10x mean_clusters (10 samples) #1'):
            repeat_mean_clusters(s1, tf.constant(10))
        with timed('run 10x mean_clusters (10 samples) #2'):
            repeat_mean_clusters(s1, tf.constant(10))
        with timed('run 10x eager mean_clusters (10 samples)'):
            for i in range(10):
                gsi.largest_clusters(s1, drop_edges=0.1, samples=10)

        gsi.print_metrics()
