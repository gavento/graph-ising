import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf
import collections


BaseVar = collections.namedtuple('BaseVar', ['variable', 'slice', 'slice_len', 'slice_name', 'orig_name'])


class Base:

    def __init__(self):
        self.ftype = np.float32
        self.itype = np.int64
        # {orig_name: BaseVar}, sorted by adding order
        self._vars = collections.OrderedDict()
        # [Keras metric object, named]
        self._metrics = []

    def _add_mean_metric(self, name):
        m = tf.keras.metrics.Mean(name)
        self._metrics.append(m)
        return m

    def _add_copy_var(self, orig_name, *, slice_len=None, capacity=None, prefix='v_', dtype=None):
        """
        Create a sliced varible proxy for `self.orig_name`.

        Updated with `_update_copy_vars`.
        The variable is either a direct proxy for `orig_name`, or can accomodate variable 
        length data with `slice_len` set to a scalar tensor and `capacity` an upper bound to slice size.
        """
        d = self.__getattribute__(orig_name, )

        if capacity is not None:
            raise NotImplementedError("Unfinished, use just the plain version")
            assert slice_len is not None
            assert isinstance(d, (np.ndarray, tf.Tensor)) and len(d.shape) > 0
            shape = list(d.shape)
            shape[0] = capacity
            var = tf.Variable(tf.zeros(shape, dtype=dtype or d.dtype), trainable=False, name=orig_name, dtype=dtype)
            slice_tensor = var[:slice_len]
            slice_tensor.assign(d)
        else:
            assert slice_len is None
            var = tf.Variable(d, trainable=False, name=orig_name, dtype=dtype)
            slice_tensor = var
        bv = BaseVar(var, slice_tensor, slice_len, prefix + orig_name, orig_name)

        self.__setattr__(bv.slice_name, bv.slice)
        self._vars[orig_name] = bv

    def _update_var(self, name_or_bv):
        if isinstance(name_or_bv, str):
            bv = self._vars[name_or_bv]
        else:
            bv = name_or_bv
        bv.slice.assign(self.__getattribute__(bv.orig_name))

    def _update_copy_vars(self):
        "Internal. Update all the TF variables with their current values."
        for bv in self._vars.values():
            self._update_var(bv)


class GraphSet(Base):

    def __init__(self, graphs):
        self._set_graphs(graphs)

        # Scalars
        self._add_copy_var('n', dtype=self.itype)
        self._add_copy_var('order', dtype=self.itype)
        self._add_copy_var('size', dtype=self.itype)
        # Per-graph
        self._add_copy_var('orders', dtype=self.itype)
        self._add_copy_var('sizes', dtype=self.itype)
        # Per-vertex
        self._add_copy_var('batch', dtype=self.itype)
        self._add_copy_var('label', dtype=self.itype)
        self._add_copy_var('in_degree', dtype=self.itype)
        self._add_copy_var('out_degree', dtype=self.itype)
        # Per-edge
        self._add_copy_var('edges', dtype=self.itype)

    def _set_graphs(self, graphs):
        """
        Assumes the nodes are numbered 0 .. order-1 (use relabel to convert on load)
        """
        if isinstance(graphs, nx.Graph):
            graphs = [graphs]
        assert len(graphs) > 0
        self.graphs = [nx.DiGraph(g) for g in graphs]
        if relabel:
            self.graphs = [nx.relabel.convert_node_labels_to_integers(g) for g in self.graphs]

        self.n = len(self.graphs)
        self.orders = np.array([g.order() for g in self.graphs], dtype=np.int64)
        self.sizes = np.array([g.size() for g in self.graphs], dtype=np.int64)
        # Joint graph size
        self.order = sum(self.orders)
        self.size = sum(self.sizes)

        # The number of graph owning a vertex
        self.batch = np.zeros([self.order], dtype=np.int64)
        # The number of vertex within original graph
        self.label = np.zeros([self.order], dtype=np.int64)
        # Array of edge starts ([0], unsorted) ond ends ([1], ascending)
        self.edges = np.zeros([2, self.size], dtype=np.int64)
        # Vertex degrees
        self.out_degrees = np.zeros([self.order], dtype=np.int64)
        self.in_degrees = np.zeros([self.order], dtype=np.int64)

        bi = 0
        ei = 0
        for gi in range(self.n):
            g = self.graphs[gi]
            # Set graph number in batch
            self.batch[bi:bi + self.orders[gi]] = gi
            # Create edges
            for vi in range(self.orders[gi]):
                self.in_degrees[bi + vi] = g.in_degree(vi)
                self.out_degrees[bi + vi] = g.out_degree(vi)
                self.label[bi + vi] = vi
                for wi in g.predecessors(vi):
                    self.edges[0, ei] = wi + bi
                    self.edges[1, ei] = vi + bi
                    ei += 1
            bi += self.orders[gi]
        assert ei == self.size
        assert bi == self.order

    def _generic_neighbors_op(self, segment_fn, node_data, edge_weights=None):
        """
        Return the segment_fn of node_data of adjacent (in-) nodes (not including self).

        Weights may be bool or numeric.
        """
        node_data = tf.identity(node_data)
        edge_data = tf.gather(node_data, self.v_edges[0])
        if edge_weights is not None:
            edge_data = tf.cast(edge_weights, node_data.dtype) * edge_data
        return segment_fn(edge_data, self.v_edges[1])

    def sum_neighbors(self, node_data, edge_weights=None):
        "Aggregate sum over (in-) neighbors."
        return self._generic_neighbors_op(tf.math.segment_sum, node_data, edge_weights=edge_weights)

    def max_neighbors(self, node_data, edge_weights=None):
        "Aggregate max over (in-) neighbors."
        return self._generic_neighbors_op(tf.math.segment_max, node_data, edge_weights=edge_weights)

    def mean_neighbors(self, node_data, edge_weights=None):
        "Aggregate mean over (in-) neighbors."
        return self._generic_neighbors_op(tf.math.segment_mean, node_data, edge_weights=edge_weights)
