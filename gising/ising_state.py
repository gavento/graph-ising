import networkx as nx
import numpy as np

from .cising import cising, ffi
from .state import State, Stats


class GraphIsingState(State):

    def __init__(self, graph, spins=None, seed=None, F=0.0, T=1.0):
        if spins is None:
            spins = np.full([graph.order()], -1, dtype='int8')
        super().__init__(spins, seed=seed)
        self.spins_up = np.sum((self.spins + 1) // 2)
        self.F = F
        self.T = T
        self.set_graph(graph)

    def set_graph(self, graph):
        """
        Set the internal graph to a list of edges [(u,v), ...].
        Every u-v edge shoul appear only once (avoid [(u,v), (v,u), ...]).
        The graph is assumed to have the same number of vertices.
        The list may also be a set, the edges may be also sets.
        """
        assert graph.order() == self.n
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.nodes_index = {k: i for i, k in enumerate(self.nodes)}

        self.degree = np.array([graph.degree[self.nodes[i]] for i in range(self.n)], dtype='int32')
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

    def _prepare_state(self):

        assert self.spins.dtype == 'int8'
        assert self.neigh_list.dtype == 'int32'
        assert self.neigh_offset.dtype == 'int32'
        assert self.degree.dtype == 'int32'
        assert self.degree_sum.dtype == 'int32'
        state = ffi.new(
            "ising_state *", {
                "n": self.n,
                "field": self.F,
                "T": self.T,
                "seed": self.seed,
                "spins_up": self.spins_up,
                "updates": self.updates,
                "spins": ffi.cast("int8_t *", self.spins.ctypes.data),
                "neigh_list": ffi.cast("int32_t *", self.neigh_list.ctypes.data),
                "neigh_offset": ffi.cast("int32_t *", self.neigh_offset.ctypes.data),
                "degree": ffi.cast("int32_t *", self.degree.ctypes.data),
                "degree_sum": ffi.cast("int32_t *", self.degree_sum.ctypes.data),
            })
        return state

    def _update_from_state(self, state):
        self.seed = state.seed
        self.spins_up = state.spins_up
        self.updates = state.updates

    def mc_updates(self, updates):
        """
        Run `updates` of randomly chosen spins (independently, with replacement),
        updating the random seed every time.

        Returns the number of flips.
        """

        assert self.neigh_list is not None

        state = self._prepare_state()
        r = cising.ising_mc_update_random(state, updates)
        self._update_from_state(state)
        return r

    def get_hamiltonian(self):
        state = self._prepare_state()
        return cising.ising_hamiltonian(state, self.F, 1.0)


class SpinCountIsingState(GraphIsingState):

    def get_order(self):
        self._order = self.spins_up
        return self.spins_up

    def _compute_stats(self):
        "Internal method to compute the stats, return `(order, stats)`. Computes spins with value 1."
        mask = (self.spins + 1) // 2
        assert np.sum(mask) == self.spins_up
        return np.sum(mask), Stats(mask)

    def update_until(self, low, high, high_tolerance=0.01, timeout=None, measure_every=1):
        "Run simulation until the order is `>=hi`, `<lo`, or for at most `timeout` MCSS."
        state = self._prepare_state()
        cising.update_until_spincount(state, low, high, int(timeout * self.n))
        self._update_from_state(state)
        self.get_stats()


class ClusterOrderIsingState(GraphIsingState):
    "Graph Ising state with the largest cluster order parameter."

    def _compute_stats(self):
        "Internal method to compute the stats."
        raise NotImplementedError()

    def update_until(self, low, high, high_tolerance=0.01, timeout=None, measure_every=1):
        "Run simulation until the order is `>=hi`, `<lo`, or for at most `timeout` MCSS."
        raise NotImplementedError()


def report_runtime_stats():

    def row(name, c, t):
        return f" {name:>30}: {c:>10}x in {t:8.3g}s ({c / max(t, 1e-32):8.3g} op/s)"

    return '\n'.join([
        row("Node updates", cising.update_count, cising.update_ns * 1e-9),
        row("Clustering node searches", cising.cluster_count, cising.cluster_ns * 1e-9),
    ])
