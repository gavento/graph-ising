import networkx as nx
import numpy as np
import tensorflow as tf


class GraphSet:
    def __init__(self, graphs, spins=-1.0, fields=0.0, temperatures=1.0, dtype=np.float32, scope='ising'):

        # Graphs and their properties
        self.graphs = list(graphs)
        self.n = len(self.graphs)
        self.orders = [g.order() for g in self.graphs]
        self.sizes = [g.size() for g in self.graphs]
        self.max_order = max(self.orders)
        self.max_size = max(self.sizes)
        self.tot_order = sum(self.orders)
        self.tot_size = sum(self.sizes)

        # Lists of original node IDs (fixing node order) and their indices
        self.nodes = [list(g.nodes()) for g in self.graphs]
        self.node_indexes = [{v: i for i, v in enumerate(nl)} for nl in self.nodes]

        # Misc parameters
        self.scope = scope
        self.dtype = dtype

        # Node attributes
        self.spins = self._nodes_array(spins)
        self.fields = self._nodes_array(fields)
        self.temperatures = self._nodes_array(temperatures)
        self.degrees = np.zeros((self.n, self.max_order), dtype=np.int32)
        self.node_mask = np.zeros((self.n, self.max_order), dtype=np.int32)

        #self.edge_starts = np.full((self.n, self.max_size), 0, dtype=np.int32)
        #self.edge_ends = np.full((self.n, self.max_order), 0, dtype=np.int32)
        # Edge starts and ends, indexes into flattened vertex list (of length n*max_order)
        # Ends are sorted, 
        self.edge_starts_global = np.zeros(self.tot_size * 2, dtype=np.int32)
        self.edge_ends_global = np.zeros(self.tot_size * 2, dtype=np.int32)
        ei = 0
        for gi, g in enumerate(self.graphs):
            for vi, v in enumerate(self.nodes[gi]):
                self.degrees[gi, vi] = g.degree[v]
                self.node_mask[gi, vi] = 1
                for w in g.neighbors(v):
                    wi = self.node_indexes[gi][w]
                    self.edge_starts_global[ei] = wi + gi * self.max_order
                    self.edge_ends_global[ei] = vi + gi * self.max_order
                    ei += 1
            self.spins[gi, g.order():] = 0.0
        assert ei == self.tot_size * 2

    def _nodes_array(self, data):
        "Helper to create / convert node data array"
        if isinstance(data, (int, float)):
            a = np.full((self.n, self.max_order), data, dtype=self.dtype)
        else:
            a = np.array(data, dtype=self.dtype)
        assert a.shape == (self.n, self.max_order)
        return a

    def construct(self):
        "All the variables still need to be initialized."
        with tf.name_scope(self.scope):
            self.v_orders = tf.Variable(self.orders, trainable=False, name='orders', dtype=tf.int32)
            self.v_sizes = tf.Variable(self.sizes, trainable=False, name='sizes', dtype=tf.int32)

            self.v_fields = tf.Variable(self.fields, trainable=False, name='fields', dtype=self.dtype)
            self.v_temperatures = tf.Variable(self.temperatures, trainable=False, name='temperatures', dtype=self.dtype)
            self.v_spins = tf.Variable(self.spins, trainable=False, name='spins', dtype=self.dtype)
            self.v_degrees = tf.Variable(self.degrees, trainable=False, name='degrees', dtype=tf.int32)
            self.v_node_mask = tf.Variable(self.node_mask, trainable=False, name='node_mask', dtype=tf.int32)
            self.v_node_mask_bool = tf.Variable(self.node_mask, trainable=False, name='node_mask_bool', dtype=tf.bool)
            self.v_edge_starts_global = tf.Variable(self.edge_starts_global, trainable=False, name='edge_starts_global', dtype=tf.int32)
            self.v_edge_ends_global = tf.Variable(self.edge_ends_global, trainable=False, name='edge_ends_global', dtype=tf.int32)

            self.metric_fraction_flipped = tf.keras.metrics.Mean('fraction_flipped')
            self.metric_mean_spin = tf.keras.metrics.Mean('mean_spin')

    def update_op(self, spins_in, update_fraction = 1.0, update_metrics=False):
        "Returns a resulting spins_out tensor operation"
        with tf.name_scope(self.scope):
            assert spins_in.shape == (self.n, self.max_order)
            sum_neighbors = self.sum_neighbors_op(spins_in)
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

    def components_op(self, iters=16):
        comp_nums = tf.reshape(tf.range(self.n * self.max_order, dtype=tf.int32), (self.n, self.max_order))
        edge_mask = None
        for i in range(iters):
            neigh_nums = self.max_neighbors_op(comp_nums, edge_mask=edge_mask)
            comp_nums = tf.maximum(comp_nums, neigh_nums)
        comp_nums = tf.reshape(comp_nums, [self.n * self.max_order])
        comp_sizes = tf.math.unsorted_segment_sum(tf.fill([self.n * self.max_order], 1), comp_nums)
        comp_sizes = tf.reshape(comp_sizes, [self.n, self.max_order])
        max_comp_sizes = tf.reduce_max(comp_sizes, axis=1)
        return max_comp_sizes


    def _neighbors_op(self, segment_fn, node_data, edge_mask=None):
        "Return the maxima of node_data of adjacent nodes (not including self)."
        assert node_data.shape == (self.n, self.max_order)
        node_data_f = tf.reshape(node_data, (self.n * self.max_order,))
        edge_data = tf.gather(node_data_f, self.v_edge_starts_global)
        if edge_mask is not None:
            assert edge_mask.shape == (self.tot_size * s,)
            edge_data = tf.cast(edge_mask, node_data.dtype) * edge_data
        node_out = segment_fn(edge_data, self.v_edge_ends_global)
        return tf.reshape(tf.pad(node_out, [[0, self.max_order - self.orders[self.n - 1]]]), (self.n, self.max_order))

    def sum_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_sum, node_data, edge_mask=edge_mask)

    def max_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_max, node_data, edge_mask=edge_mask)

    def mean_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_mean, node_data, edge_mask=edge_mask)

    def sum_neighbors_op_2(self, node_data):
        "Return the maxima of node_data of adjacent nodes (not including self). (optimized fn?)"
        assert node_data.shape == (self.n, self.max_order)
        node_data_f = tf.reshape(node_data, (self.n * self.max_order,))
        edge_data = tf.gather(node_data_f, self.v_edge_starts_global)
        node_out = tf.math.segment_sum(edge_data, self.v_edge_ends_global)
        return tf.reshape(tf.pad(node_out, [[0, self.max_order - self.orders[self.n - 1]]]), (self.n, self.max_order))
