import copy
import random

import attr
import numpy as np
import tqdm


@attr.s
class Stats:
    mask = attr.ib()


class State:

    def __init__(self, spins, seed=None):
        self.spins = np.array(spins, dtype='int8')
        assert len(self.spins.shape) == 1
        self.n = len(self.spins)
        self.seed = random.randint(0, (1 << 63) - 1) if seed is None else seed
        self._stats = None
        self._order = None
        self.updates = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.n} spins order {self._order if self._order is not None else np.nan:.3g}>"

    def copy(self):
        """
        A shallow copy of the class (sharing any graph etc.), creating own copy of the spins.
        """
        c = copy.copy(self)
        c.spins = self.spins.copy()
        return c

    def get_stats(self):
        "Return the stats associated to the state, optionally computing and caching them."
        if self._stats is None:
            self._order, self._stats = self._compute_stats()
        return self._stats

    def get_order(self):
        "Return the stats associated to the state, optionally computing and caching them."
        if self._order is None:
            self._order, self._stats = self._compute_stats()
        return self._order

    def _compute_stats(self):
        "Internal method to compute the stats, return `(order, stats)`."
        raise NotImplementedError()

    def update_until(self, low, high, high_tolerance=0.01, timeout=None, measure_every=1):
        "Run simulation until the order is `>=hi`, `<lo`, or for at most `timeout` MCSS."
        raise NotImplementedError()

    def sample_mesostable(self, progress=True, time=200, samples=100, trials=5):
        r = range(trials)
        if progress:
            r = tqdm.tqdm(r,
                          "Sampling mesostable",
                          file=progress if progress is not True else sys.stderr)
        spl = np.zeros((trials, samples))
        for ti in r:
            state = self.copy()
            state.seed = np.random.randint(1 << 60)
            for si in range(samples):
                state.mc_updates(max(int(time / samples * state.n), 1))
                spl[ti, si] = state.get_order()
        return spl
