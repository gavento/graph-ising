import time

import attr
import numpy as np
import tensorflow.compat.v2 as tf

from .cising import IsingState


@attr.s
class PopSample:
    param = attr.ib(type=float)
    interface_no = attr.ib(type=int)
    parent = attr.ib(type='PopSample', repr=False)
    time = attr.ib(type=float)
    state = attr.ib(type=IsingState, repr=False)
    #spins = attr.ib(type=np.ndarray, repr=False)

    sampled = attr.ib(0)
    up_times = attr.ib(factory=list)
    down_times = attr.ib(factory=list)
    timeouts = attr.ib(factory=list)


@attr.s(repr=False)
class Interface:
    param = attr.ib(type=float)
    pops = attr.ib(factory=list, repr=False)

    def up_times(self):
        if not self.pops:
            return np.zeros(0)
        return np.concatenate([p.up_times for p in self.pops])

    def energies(self):
        return np.array([p.state.get_hamiltonian() for p in self.pops])

    def down_times(self):
        if not self.pops:
            return np.zeros(0)
        return np.concatenate([p.down_times for p in self.pops])

    def timeouts(self):
        if not self.pops:
            return np.zeros(0)
        return np.concatenate([p.timeouts for p in self.pops])

    def get_random_pop(self):
        assert len(self.pops) > 0
        return self.pops[np.random.randint(0, len(self.pops))]

    def __repr__(self):
        uts = self.up_times()
        dts = self.down_times()
        tos = self.timeouts()
        mut = np.mean(uts) if len(uts) > 0 else 0.0
        mdt = np.mean(dts) if len(dts) > 0 else 0.0
        tot = len(uts) + len(dts)
        frac = len(uts) / tot if tot > 0 else 0.0
        return "{}(param={}, {} pops, {} ups (time {:.2g}), {} downs (time {:.2g})), {} TOs, {:.4f} up".format(
            self.__class__.__name__, self.param, len(self.pops), len(uts), mut, len(dts), mdt, len(tos), frac)


class FFSampler:

    def __init__(self, graph, interfaces, min_pop_size=10, cluster_samples=1, cluster_e_prob=1.0):
        self.graph = graph
        self.min_pop_size = min_pop_size
        self.cluster_samples = cluster_samples
        self.cluster_e_prob = cluster_e_prob

        self.ran_updates = 0.0
        self.ran_clusters = 0

        self.interfaces = [i if isinstance(i, Interface) else Interface(i) for i in interfaces]

    def fill_interfaces(self):
        for ino, iface in enumerate(self.interfaces[:-1]):
            next_iface = self.interfaces[ino + 1]
            its = 0
            t0 = time.perf_counter()
            while len(next_iface.pops) < self.min_pop_size:
                its += 1
                self.trace_from_iface_dyn(ino)
            t1 = time.perf_counter()
            print("{} iters from iface {} finished in {} s:\n  iface {:3d}: {}\n  iface {:3d}: {}".
                  format(its, ino, t1 - t0, ino, iface, ino + 1, next_iface))


class CIsingFFSampler(FFSampler):

    def __init__(self, graph, interfaces, state=None, **kwargs):
        super().__init__(graph, interfaces, **kwargs)
        if state is None:
            state = IsingState(graph=graph)
        self.interfaces[0].pops.append(PopSample(self.interfaces[0].param, 0, None, 0.0, state))

    def trace_from_iface_dyn(self, ino, max_time=100.0):

        iface = self.interfaces[ino]
        up_iface = self.interfaces[ino + 1]
        down_iface = self.interfaces[max(ino - 1, 0)]
        up_param = up_iface.param
        down_param = down_iface.param  # special case for ino == 0

        cluster_every = 0.02 # TODO dynamic autotune

        pop = iface.get_random_pop()
        pop.sampled += 1
        state = pop.state.copy()
        state.seed += np.random.randint(1<<32)
        t = 0.0  # Time

        while True:
            updates = max(1, int(cluster_every * state.n))
            dt = updates / state.n
            state.mc_sweep(sweeps=0, updates=updates)
            t += dt
            self.ran_updates += dt

            cstats = state.mc_max_cluster(samples=self.cluster_samples, edge_prob=self.cluster_e_prob)
            param = cstats.v_in
            self.ran_clusters += 1

            if param >= up_param:
                pop.up_times.append(t)
                npop = PopSample(param, ino + 1, pop, t, state)
                up_iface.pops.append(npop)
                break

            if param <= down_param and ino > 0:
                pop.down_times.append(t)
                npop = PopSample(param, ino - 1, pop, t, state)
                down_iface.pops.append(npop)
                break

            if t > max_time:
                pop.timeouts.append(t)
                print("TO", t)
                break
