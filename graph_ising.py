import networkx as nx
import numpy as np
import tensorflow as tf

from tfgraph import TFGraph
from utils import timed


class GraphIsing:
    def __init__(self, graphs_or_n, max_order=None, max_size=None, scope='ising'):
        if isinstance(graphs_or_n, int):
            self.n = graphs_or_n
            assert (max_order is not None) and (max_size is not None)
        else:
            graphs_or_n = list(graphs_or_n)
            self.n = len(graphs_or_n)
            if max_order is None:
                max_order = max(g.order for g in graphs_or_n)
            if max_size is None:
                max_size = max(g.size for g in graphs_or_n)

        self.max_order = max_order
        self.max_size = max_size
        self.scope = scope
        self.dtype = np.float32

        # Never use these in the computations - use only consts above and variables below
        self.empty_graph = TFGraph(None, max_order, max_size)
        self.graphs = [self.empty_graph] * self.n

        def vf(shape, val, name, t=self.dtype):
            "Helper creating a const variable of given shape"
            return tf.Variable(np.full(shape, val, dtype=t), trainable=False, name=name, dtype=t)

        with tf.name_scope(self.scope):
            self.v_orders = vf([self.n], 0, 'orders', np.int64)
            self.v_sizes = vf([self.n], 0, 'sizes', np.int64)
            self.v_tot_order = vf([], 0, 'tot_order', np.int64)
            self.v_tot_size = vf([], 0, 'tot_size', np.int64)
            self.v_fields = vf([self.n, self.max_order], 0.0, 'fields')
            self.v_temperatures = vf([self.n, self.max_order], 0.0, 'temperatures')
            self.v_degrees = vf([self.n, self.max_order], 0, 'degrees', np.int64)
            self.v_node_masks = vf([self.n, self.max_order], False, 'node_mask', bool)
            # Edge starts numbered within every graph independently
            self.v_edge_starts = vf([self.n, self.max_size * 2], 0, 'edge_starts', np.int64)
            self.v_edge_ends = vf([self.n, self.max_size * 2], 0, 'edge_starts', np.int64)
            # Edge starts numbered globally (by flattened node indices)
            self.v_edge_starts_global = vf(self.n * self.max_size * 2, 0, 'edge_starts_global', np.int64)
            self.v_edge_ends_global = vf(self.n * self.max_size * 2, 0, 'edge_starts_global', np.int64)
            # Update op metrics
            self.metric_fraction_flipped = tf.keras.metrics.Mean('fraction_flipped')
            self.metric_mean_spin = tf.keras.metrics.Mean('mean_spin')

        if not isinstance(graphs_or_n, int):
            self.set_graphs(graphs_or_n)

    def set_graphs(self, graphs):
        "Use given TFGraphs reusing the same variables and computation graph"
        graphs = list(graphs)
        assert all(isinstance(g, TFGraph) for g in graphs)
        assert all(g.max_order <= self.max_order for g in graphs)
        assert all(g.max_size <= self.max_size for g in graphs)
        assert len(graphs) <= self.n
        graphs.extend([self.empty_graph] * (self.n - len(graphs)))
        self.graphs = graphs

        self.v_orders.assign([g.order for g in graphs])
        self.v_sizes.assign([g.size for g in graphs])
        self.v_tot_order.assign(sum(g.order for g in graphs))
        self.v_tot_size.assign(sum(g.size for g in graphs))

        def varfill(var, d, l):
            var.assign(np.stack([np.pad(a, [(0, l - len(a))], 'constant') for a in d]))

        varfill(self.v_fields, [g.fields for g in graphs], self.max_order)
        varfill(self.v_temperatures, [g.temperatures for g in graphs], self.max_order)
        varfill(self.v_degrees, [g.degrees for g in graphs], self.max_order)
        varfill(self.v_node_masks, [g.node_mask for g in graphs], self.max_order)
        varfill(self.v_edge_starts, [g.edge_starts for g in graphs], self.max_size * 2)
        varfill(self.v_edge_ends, [g.edge_ends for g in graphs], self.max_size * 2)

        glob_starts = np.zeros(self.n * self.max_size * 2, dtype=np.int64)
        glob_ends = np.zeros(self.n * self.max_size * 2, dtype=np.int64)
        ei = 0
        for gi, g in enumerate(graphs):
            k = g.size * 2
            glob_starts[ei:ei + k] = g.edge_starts[:k] + (gi * self.max_order)
            glob_ends[ei:ei + k] = g.edge_ends[:k] + (gi * self.max_order)
            ei += k
        assert ei == self.v_tot_size.numpy() * 2
        self.v_edge_starts_global.assign(glob_starts)
        self.v_edge_ends_global.assign(glob_ends)

    def initial_spins(self, value=-1.0):
        "Returns initial spin values as `tf.constant`."
        return tf.constant(np.stack([g.initial_spins(self.max_order, value) for g in self.graphs]), dtype=self.dtype)

    def _neighbors_op(self, segment_fn, node_data, edge_mask=None):
        "Return the maxima of node_data of adjacent nodes (not including self)."
        with tf.name_scope(self.scope):
            assert node_data.shape == (self.n, self.max_order)
            node_data_f = tf.reshape(node_data, (self.n * self.max_order,))
            k = self.v_tot_size * 2
            edge_data = tf.gather(node_data_f, self.v_edge_starts_global[:k])
            if edge_mask is not None:
                edge_data = tf.cast(edge_mask, node_data.dtype) * edge_data
            node_out = segment_fn(edge_data, self.v_edge_ends_global[:k])
            pad_out = self.n * self.max_order - tf.shape(node_out)[0]
            return tf.reshape(tf.pad(node_out, [[0, pad_out]]), (self.n, self.max_order))

    def sum_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_sum, node_data, edge_mask=edge_mask)

    def max_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_max, node_data, edge_mask=edge_mask)

    def mean_neighbors_op(self, node_data, edge_mask=None):
        return self._neighbors_op(tf.math.segment_mean, node_data, edge_mask=edge_mask)

    def update_op(self, spins_in, update_fraction=1.0, update_metrics=True):
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
                self.metric_fraction_flipped.update_state(tf.cast(flip, self.dtype), update & self.v_node_masks)
                self.metric_mean_spin.update_state(spins_out, self.v_node_masks)
            return spins_out

    def sampled_largest_component_size_op(self, spins, iters=tf.constant(16), drop_edges=tf.constant(0.5), drop_samples=tf.constant(10), positive_spin=True):
        mean_max_sizes = tf.zeros([self.n], dtype=self.dtype)
        for i in range(drop_samples):
            mean_max_sizes += tf.cast(self.largest_component_size_op(spins, iters, drop_edges, positive_spin), self.dtype)
        return mean_max_sizes / tf.cast(drop_samples, self.dtype)

    @tf.function
    def largest_component_size_op(self, spins, iters=tf.constant(16), drop_edges=None, positive_spin=True):
        with tf.name_scope(self.scope):
            K = self.n * self.max_order
            if positive_spin:
                node_mask = tf.cast(spins > 0.0, tf.int64)
            else:
                node_mask = tf.cast(spins < 0.0, tf.int64)
            if drop_edges is not None:
                lls = tf.math.log([[drop_edges, 1.0 - drop_edges]])
                edge_mask = tf.reshape(tf.random.categorical(lls, tf.cast(self.v_tot_size, tf.int32) * 2), (-1, ))
            else:
                edge_mask = None

            comp_nums = tf.reshape(tf.range(1, K + 1, dtype=tf.int64), (self.n, self.max_order))
            comp_nums = comp_nums * node_mask
            for i in range(iters):
                neigh_nums = self.max_neighbors_op(comp_nums, edge_mask=edge_mask)
                comp_nums = tf.maximum(comp_nums, neigh_nums)
                comp_nums = comp_nums * node_mask
            comp_nums = tf.reshape(comp_nums, [K])

            comp_sizes = tf.math.unsorted_segment_sum(tf.fill([K], 1), comp_nums, K + 1)
            comp_sizes = comp_sizes[1:]  # drop the 0-component
            comp_sizes = tf.reshape(comp_sizes, [self.n, self.max_order])

            max_comp_sizes = tf.reduce_max(comp_sizes, axis=1)
            return max_comp_sizes
