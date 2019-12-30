import networkx as nx
import numpy as np

from .cising import cising, ffi
from .state import State, Stats, GraphState


class GraphIsingState(GraphState):

    def __init__(self, graph, spins=None, seed=None, F=0.0, T=1.0):
        super().__init__(graph, spins=spins, seed=seed)
        self.spins_up = np.sum((self.spins + 1) // 2)
        self.F = F
        self.T = T

    def _prepare_state(self):

        state = ffi.new(
            "ising_state *", {
                "n": self.n,
                "field": self.F,
                "T": self.T,
                "seed": self.seed,
                "spins_up": self.spins_up,
                "updates": self.updates,
                "spins": ffi.cast("int8_t *", self.spins.ctypes.data),
                "g": self.cgraph.get_ising_graph(),
            })
        return state

    def _update_from_state(self, state):
        self.seed = state.seed
        self.spins_up = state.spins_up
        self.updates = state.updates
        self._stats = None
        self._order = state.spins_up

    def mc_updates(self, updates):
        """
        Run `updates` of randomly chosen spins (independently, with replacement),
        updating the random seed every time.

        Returns the number of flips.
        """

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


class ClusterOrderIsingState(GraphIsingState):
    "Graph Ising state with the largest cluster order parameter."

    def _compute_stats(self):
        "Internal method to compute the stats."
        raise NotImplementedError()

    def update_until(self, low, high, high_tolerance=0.01, timeout=None, measure_every=1):
        "Run simulation until the order is `>=hi`, `<lo`, or for at most `timeout` MCSS."
        self._stats = None
        self._order = None
        raise NotImplementedError()


def report_runtime_stats():

    def row(name, c, t):
        return f" {name:>30}: {c:>10}x in {t:8.3g}s ({c / max(t, 1e-32):8.3g} op/s)"

    return '\n'.join([
        row("Node updates", cising.update_count, cising.update_ns * 1e-9),
        row("Clustering node searches", cising.cluster_count, cising.cluster_ns * 1e-9),
    ])
