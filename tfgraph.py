import networkx as nx
import numpy as np
import tensorflow as tf


class TFGraph:
    def __init__(self, graph, max_order=None, max_size=None, fields=0.0, temperatures=1.0):
        self.dtype = np.float32
        self.constructed = False

        # Graph and its nodes
        if graph is None:
            graph = nx.Graph()
        self.graph = graph
        self.order = self.graph.order()
        self.size = self.graph.size()
        self.max_size = max_size or self.size
        assert self.max_size >= self.size
        self.max_order = max_order or self.order
        assert self.max_order >= self.order
        
        # Per-node attributes
        self.fields = self._nodes_array(fields)
        self.temperatures = self._nodes_array(temperatures)
        self.degrees = np.zeros(self.max_order, dtype=np.int32)
        self.node_mask = np.zeros(self.max_order, dtype=bool)
        # Lists of original node IDs (fixing node order) and their indices
        self.nodes = sorted(list(self.graph.nodes()))
        self.node_index = {v: i for i, v in enumerate(self.nodes)}
        self.adj_lists = [[self.node_index[w] for w in self.graph.neighbors(v)] for v in self.nodes]
        # The ends are sorted in ascending order
        self.edge_starts = np.zeros(self.max_size * 2, dtype=np.int32)
        self.edge_ends = np.zeros(self.max_size * 2, dtype=np.int32)
        ei = 0
        for vi in range(self.order):
            self.degrees[vi] = len(self.adj_lists[vi])
            self.node_mask[vi] = True
            for wi in self.adj_lists[vi]:
                self.edge_starts[ei] = wi
                self.edge_ends[ei] = vi
                ei += 1
        assert ei == self.size * 2

    def initial_spins(self, size, value=-1.0):
        "Returns initial spin values (padded to `size`) as `tf.constant`."
        spins = np.full(size, value)
        spins[self.order:] = 0.0
        return tf.constant(spins, dtype=self.dtype)

    def _nodes_array(self, data):
        "Helper to create / convert node data array"
        if isinstance(data, (int, float)):
            return np.full(self.max_order, data, dtype=self.dtype)
        else:
            return np.array(data, dtype=self.dtype)

