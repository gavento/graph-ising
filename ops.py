import tensorflow as tf
import numpy as np
import networkx as nx


class GraphSet:
    def __init__(self, graphs, spins=-1.0, fields=0.0, temperatures=1.0, sparse=False, dtype=np.float32, scope='ising'):
        self.scope = scope
        if isinstance(graphs, nx.Graph):
            graphs = [graphs]
        self.graphs = list(graphs)
        # Original node IDs in the selected order (may be customized e.g. for caching)
        self.node_lists = [list(g.nodes()) for g in self.graphs]
        self.node_indexes = [{v: i for i, v in enumerate(nl)} for nl in self.node_lists]

        self.n = len(self.graphs)
        self.orders = [len(nl) for nl in self.node_lists]
        self.max_order = max(self.orders)
        self.dtype = dtype

        def vertices_array(data):
            if isinstance(data, (int, float)):
                a = np.full((self.n, self.max_order), data, dtype=self.dtype)
            else:
                a = np.array(data, dtype=dtype)
            assert a.shape == (self.n, self.max_order)
            return a

        self.spins = vertices_array(spins)
        self.fields = vertices_array(fields)
        self.temperatures = vertices_array(temperatures)
        self.degrees = np.full((self.n, self.max_order), 0, dtype=np.int32)
        self.node_mask = np.full((self.n, self.max_order), 0, dtype=np.int32)

        for gi, g in enumerate(self.graphs):
            for vi, v in enumerate(g.nodes()):
                self.degrees[gi, vi] = g.degree[v]
                self.node_mask[gi, vi] = 1
            self.spins[gi, g.order():] = 0.0

        self.sparse = bool(sparse)
        if self.sparse:
            raise NotImplementedError
        else:
            self.adj_m = np.zeros((self.n, self.max_order, self.max_order), dtype=dtype)
            for gi, g in enumerate(self.graphs):
                for vi, v in enumerate(self.node_lists[gi]):
                    for w in g.neighbors(v):
                        wi = self.node_indexes[gi][w]
                        self.adj_m[gi, vi, wi] = 1
                        # Likely not needed due to symmetry
                        self.adj_m[gi, wi, vi] = 1

    def construct(self):
        "All the variables need to be initialized."
        with tf.name_scope(self.scope):
            self.v_fields = tf.Variable(self.fields, trainable=False, name='fields', dtype=self.dtype)
            self.v_temperatures = tf.Variable(self.temperatures, trainable=False, name='temperatures', dtype=self.dtype)
            self.v_spins = tf.Variable(self.spins, trainable=False, name='spins', dtype=self.dtype)
            self.v_degrees = tf.Variable(self.degrees, trainable=False, name='degrees', dtype=tf.int32)
            self.v_node_mask = tf.Variable(self.node_mask, trainable=False, name='node_mask', dtype=tf.int32)
            self.v_node_mask_bool = tf.Variable(self.node_mask, trainable=False, name='node_mask', dtype=tf.bool)
            self.v_orders = tf.Variable(self.orders, trainable=False, name='orders', dtype=tf.int32)
            if self.sparse:
                raise NotImplementedError
            else:
                self.v_adj_m = tf.Variable(self.adj_m, trainable=False, name='adj_m', dtype=self.dtype)
            self.metric_fraction_flipped = tf.keras.metrics.Mean('fraction_flipped')
            self.metric_mean_spin = tf.keras.metrics.Mean('mean_spin')

    def update_op(self, spins_in, update_fraction = 1.0, update_metrics=False):
        "Returns a resulting spins_out tensor operation"
        with tf.name_scope(self.scope):
            assert spins_in.shape == (self.n, self.max_order)
            if self.sparse:
                raise NotImplementedError
            else:
                sum_neighbors = tf.linalg.matvec(self.adj_m, spins_in)
            # delta energy if flipped
            delta_E = (self.v_fields - sum_neighbors) * (1 + 2 * spins_in) # TODO: Likely wrong second half
            # probability of random flip
            random_flip = tf.random.uniform((self.n, self.max_order), name='flip_p') < tf.math.exp(delta_E / self.v_temperatures)
            # updating only a random subset
            update = (tf.random.uniform((self.n, self.max_order), name='update_p') < update_fraction)
            # combined condition
            flip = ((delta_E > 0.0) | (random_flip)) & update
            # update spins
            spins_out = tf.where(flip, -spins_in, spins_in)
            # metrics
            if update_metrics:
                op1 = self.metric_fraction_flipped.update_state(tf.cast(flip, self.dtype), update & self.v_node_mask_bool)
                op2 = self.metric_mean_spin.update_state(spins_out, self.v_node_mask)
                with tf.control_dependencies([op1, op2]):
                    spins_out = tf.identity(spins_out)
            return spins_out

    def neighbors_max_op(self, node_data, edge_mask=None):
        "Return the maxima of node_data of adjacent nodes"
        assert node_data.shape == (self.n, self.max_order)
        edge_data = tf.gather(node_data, self.v_edge_starts, axis=1)
        if edge_mask is not None:
            assert edge_mask.shape == (self.n, self.max_edges)
            edge_data = tf.cast(edge_mask, node_data.dtype) * edge_data
        edge_data_reshaped = tf.reshape(edge_data, (self.n * self.max_order, self.max_edges))
        node_out = tf.math.segment_max(edge_data_reshaped, self.v_edge_ends_global)
        return tf.reshape(tf.pad(node_out, [[0, self.max_order - self.orders[self.n - 1]]]), (self.n, self.max_order))

import time
import contextlib

@contextlib.contextmanager
def timed(name=None):
    t0 = time.time()
    yield
    t1 = time.time()
    print((name + " " if name else "") + "took {:.3f} s".format(t1 - t0))

@tf.function
def repeated(s):
    for i in range(10):
        s = gs.update_op(s, 1.0, True)
    return s

g = nx.complete_graph(100)
gs = GraphSet([g] * 1000)
gs.construct()

s2 = gs.spins
with timed("iters"):
    for i in range(10):
        s2 = gs.update_op(s2)

with timed("wrapped 1. call"):
    s2 = repeated(gs.spins)
with timed("wrapped 2. call"):
    s2 = repeated(gs.spins)

with timed("create GS"):
    gs2 = GraphSet([g] * 1000)
with timed("construct GS"):
    gs2.construct()

with timed("other wrapped 1. call"):
    s2 = repeated(gs.spins)
with timed("other wrapped 2. call"):
    s2 = repeated(gs.spins)
