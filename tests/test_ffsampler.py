import networkx as nx
import numpy as np
import pytest
import tensorflow.compat.v2 as tf

from graph_ising.forward_flux import DirectIsingClusterFFSampler
from graph_ising.utils import get_device, timed

#from tensorflow.python.ops import control_flow_util



#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
#tf.autograph.set_verbosity(0, alsologtostdout=True)

@pytest.mark.slow
def test_basic():
    K = 10  # Graphs
    N = 100  # Nodes per graph
    g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    ff = DirectIsingClusterFFSampler(g, [0, 10, 20, 40, 60, 80, 95], batch_size=K, T=4.0)
    with timed('warmup'):
        ff.run_batch(0, 0)
    with timed('10000 steps, 0.1%% clusters'):
        ff.run_batch(0, 1000)
