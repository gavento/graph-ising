import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf

from .base import Base
from .utils import timed

class GraphSet(Base):

    def __init__(self, *, graphs=(), **kwargs):
        super().__init__(**kwargs)
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
        self._add_copy_var('in_degrees', dtype=self.itype)
        self._add_copy_var('out_degrees', dtype=self.itype)
        # Per-edge
        self._add_copy_var('edges', dtype=self.itype)

    def _set_graphs(self, graphs):
        """
        Assumes the nodes are numbered 0 .. order-1 (use relabel to convert on load)
        """
        if isinstance(graphs, nx.Graph):
            graphs = [graphs]
        assert len(graphs) > 0
        assert all(isinstance(g, nx.Graph) for g in graphs)
        self.graphs = graphs

        self.n = len(self.graphs)
        self.orders = np.array([g.order() for g in self.graphs], dtype=np.int64)
        self.sizes = np.array([g.size() if isinstance(g, nx.DiGraph) else g.size() * 2 for g in self.graphs], dtype=np.int64)
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
                self.label[bi + vi] = vi
                if isinstance(g, nx.DiGraph):
                    self.in_degrees[bi + vi] = g.in_degree(vi)
                    self.out_degrees[bi + vi] = g.out_degree(vi)
                    preds = g.predecessors(vi)
                else:
                    self.in_degrees[bi + vi] = g.degree(vi)
                    self.out_degrees[bi + vi] = g.degree(vi)
                    preds = g.neighbors(vi)
                for wi in preds:
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
