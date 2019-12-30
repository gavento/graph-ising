import networkx as nx
import numpy as np

from .cising import cising, ffi


class IsingGraph:

    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.n = len(self.nodes)
        self.nodes_index = {k: i for i, k in enumerate(self.nodes)}

        self.degree = np.array([self.graph.degree[self.nodes[i]] for i in range(self.n)],
                               dtype='int32')
        self.degree_sum = np.cumsum(self.degree, dtype='int32')
        self.neigh_offset = np.array(np.concatenate([[0], self.degree_sum[:-1]]), dtype='int32')
        self.neigh_list = np.full([self.degree_sum[-1]], -1, dtype='int32')

        offsets = self.neigh_offset.copy()
        for u, v in self.graph.edges:
            ui = self.nodes_index[u]
            vi = self.nodes_index[v]
            self.neigh_list[offsets[ui]] = vi
            offsets[ui] += 1
            self.neigh_list[offsets[vi]] = ui
            offsets[vi] += 1
        assert all(offsets == self.degree_sum)

        self._ising_graph = self._get_ising_graph()

    def get_ising_graph(self):
        return self._ising_graph

    def _get_ising_graph(self):
        assert self.neigh_list.dtype == 'int32'
        assert self.neigh_offset.dtype == 'int32'
        assert self.degree.dtype == 'int32'
        assert self.degree_sum.dtype == 'int32'

        return ffi.new(
            "ising_graph *", {
                "n": self.n,
                "neigh_list": ffi.cast("int32_t *", self.neigh_list.ctypes.data),
                "neigh_offset": ffi.cast("int32_t *", self.neigh_offset.ctypes.data),
                "degree": ffi.cast("int32_t *", self.degree.ctypes.data),
                #"degree_sum": ffi.cast("int32_t *", self.degree_sum.ctypes.data),
            })

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, repr(self.graph))
