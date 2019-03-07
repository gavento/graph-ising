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
        self.metrics = []

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
            def new_mmetric(name):
                m = tf.keras.metrics.Mean(name)
                self.metrics.append(m)
                return m

            self.metric_update_flipped = new_mmetric('update/flipped')
            self.metric_update_mean_spin = new_mmetric('update/mean_spin')
            self.metric_components_unfinished = new_mmetric('components/unfinished')
            self.metric_components_iterations = new_mmetric('components/iterations')

        if not isinstance(graphs_or_n, int):
            self.set_graphs(graphs_or_n)

    def log_metrics(self):
        for m in self.metrics:
            s = "{:20s}: {:.3g}".format(m.name, m.result().numpy())
            print(s)  # Todo: get more fancy

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

    def _generic_neighbors_op(self, segment_fn, node_data, edge_mask=None):
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

    def sum_neighbors(self, node_data, edge_mask=None):
        return self._generic_neighbors_op(tf.math.segment_sum, node_data, edge_mask=edge_mask)

    def max_neighbors(self, node_data, edge_mask=None):
        return self._generic_neighbors_op(tf.math.segment_max, node_data, edge_mask=edge_mask)

    def mean_neighbors(self, node_data, edge_mask=None):
        return self._generic_neighbors_op(tf.math.segment_mean, node_data, edge_mask=edge_mask)

    def update(self, spins_in, update_fraction=1.0, update_metrics=True):
        "Returns a resulting spins_out tensor operation"
        assert spins_in.shape == (self.n, self.max_order)
        sum_neighbors = self.sum_neighbors(spins_in)
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
        self.metric_update_flipped.update_state(tf.cast(flip, self.dtype), update & self.v_node_masks)
        self.metric_update_mean_spin.update_state(spins_out, self.v_node_masks)
        return spins_out

    def find_components(self, node_mask=None, edge_mask=None, max_iters=32):
        """
        Return component numbers for every vertex as shape [n, order].
        Components are numbered 1 .. (n * max_order), comp 0 is used for masked and
        non-existing vertices.
        """

        K = self.n * self.max_order
        initial_comp_nums = tf.reshape(tf.range(1, K + 1, dtype=tf.int64), (self.n, self.max_order))
        if node_mask is not None:
            initial_comp_nums = initial_comp_nums * node_mask

        comp_nums = tf.Variable(initial_comp_nums, dtype=tf.int64)
        iters = tf.Variable(0, dtype=tf.int64)
        unfinished = tf.Variable(0.0, dtype=self.dtype)
        run = tf.Variable(True)

        while run:
            neigh_nums = self.max_neighbors(comp_nums, edge_mask=edge_mask)
            mask_neigh_nums = neigh_nums if node_mask is None else neigh_nums * tf.cast(node_mask, tf.int64)
            new_comp_nums = tf.maximum(comp_nums, mask_neigh_nums)
            iters.assign_add(1)
            if tf.reduce_all(tf.equal(new_comp_nums, comp_nums)):
                run.assign(False)
            comp_nums.assign(new_comp_nums)
            if iters >= tf.cast(max_iters, tf.int64):
                unfinished.assign(1.0)
                run.assign(False)

        self.metric_components_unfinished.update_state(unfinished)
        self.metric_components_iterations.update_state(tf.cast(iters, tf.float32))

        if node_mask is not None:
            comp_nums == comp_nums * tf.cast(node_mask, tf.int64)
        return tf.identity(comp_nums)

    def largest_cluster(self, spins, positive_spin=True, drop_edges=None, max_iters=32):
        """
        Return the size of larges positive (res. negative) spin cluster for every graph.
        If given, `drop_edges`-fraction of edges is ignored.
        """
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

        comp_nums = self.find_components(node_mask=node_mask, edge_mask=edge_mask, max_iters=max_iters)
        comp_nums = tf.reshape(comp_nums, [K])
        comp_sizes = tf.math.unsorted_segment_sum(tf.fill([K], 1), comp_nums, K + 1)
        comp_sizes = comp_sizes[1:]  # drop the 0-th component
        comp_sizes = tf.reshape(comp_sizes, [self.n, self.max_order])

        max_comp_sizes = tf.reduce_max(comp_sizes, axis=1)
        return max_comp_sizes

    def sampled_largest_cluster(self, spins, positive_spin=True, drop_edges=tf.constant(0.5), samples=tf.constant(10), max_iters=32):
        """
        Mean largest_cluster over several samples.
        """
        mean_max_sizes = tf.Variable(tf.zeros([self.n], dtype=self.dtype))
        for i in range(samples):
            largest = self.largest_cluster(spins, positive_spin=positive_spin, drop_edges=drop_edges, max_iters=max_iters)
            mean_max_sizes.assign_add(tf.cast(largest, self.dtype))
        return mean_max_sizes / tf.cast(samples, self.dtype)



