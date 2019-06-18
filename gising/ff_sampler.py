import time
import tqdm
import attr
import numpy as np
import tensorflow.compat.v2 as tf

from .cising import IsingState, ClusterStats


@attr.s
class PopSample:
    param = attr.ib(type=float)
    interface_no = attr.ib(type=int)
    parent = attr.ib(type='PopSample', repr=False)
    time = attr.ib(type=float)
    state = attr.ib(type=IsingState, repr=False)
    cluster_stats = attr.ib(type=ClusterStats, repr=False)

    sampled = attr.ib(0)
    up_times = attr.ib(factory=list)
    down_times = attr.ib(factory=list)
    timeouts = attr.ib(factory=list)


@attr.s(repr=False)
class Interface:
    param = attr.ib(type=float)
    number = attr.ib(type=int)
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
            self.__class__.__name__, self.param, len(self.pops), len(uts), mut, len(dts), mdt,
            len(tos), frac)


class FFSampler:

    def __init__(self, graph, interfaces, min_pop_size=10, cluster_samples=1, cluster_e_prob=1.0):
        self.graph = graph
        self.min_pop_size = min_pop_size
        self.cluster_samples = cluster_samples
        self.cluster_e_prob = cluster_e_prob

        self.ran_updates = 0.0
        self.ran_clusters = 0

        self.interfaces = [
            iface if isinstance(iface, Interface) else Interface(iface, i)
            for i, iface in enumerate(interfaces)
        ]
        self.start_pop = None
        self.ifaceA = self.interfaces[0]
        self.ifaceB = self.interfaces[-1]

    def fill_interfaces(self, progress=False, timeout=100.0, tgt_samples=10, speriod_prior=0.1):
        speriod = speriod_prior
        min_samples = tgt_samples / 2
        for ino, iface in enumerate(self.interfaces):
            bot = self.ifaceA if ino > 0 else None
            Us, Ds, TOs, TUNs, As = [], [], [], [], []
            if progress:
                pb = tqdm.tqdm(range(self.min_pop_size),
                               f"Gen interface {ino} [{iface.param:6.2f}]")
            while len(iface.pops) < self.min_pop_size:
                if ino == 0:
                    pop = self.start_pop
                else:
                    pop = self.interfaces[ino - 1].get_random_pop()

                res, time, samples, npop = self.trace_from_iface_dyn(
                    pop, bot, iface, timeout=timeout, speriod=speriod, min_samples=min_samples)

                if samples >= min_samples:
                    [TOs, Us, Ds][res].append(time)
                else:
                    TUNs.append(time)
                As.append(time)
                speriod = max(np.quantile(As, 0.2) / tgt_samples, 0.02) 
                if progress:
                    pb.update(len(iface.pops) - pb.n)
                    pb.set_postfix_str(
                        f"{len(Us)} U, {len(Ds)} D, {len(TOs)} TO, {len(TUNs)} TUN, {speriod:.3g} swp/sample"
                    )
                    pb.display()
            if progress:
                pb.close()
            print(f"  genrated {ino} / {len(self.interfaces)}" +
                  f", Param mean {np.mean([p.param for p in iface.pops]):.3g} tgt {iface.param:.3g}" + 
                  (f", UpTime: mean {np.mean(Us):.3g}, min {np.min(Us):.3g}" if Us else "") +
                  (f", DownTime: mean {np.mean(Ds):.3g}, min {np.min(Ds):.3g}" if Ds else "")) 


class CIsingFFSampler(FFSampler):

    def __init__(self, graph, interfaces, state=None, **kwargs):
        super().__init__(graph, interfaces, **kwargs)
        if state is None:
            state = IsingState(graph=graph)
        self.start_pop = PopSample(0.0, -1, None, 0.0, state, None)

    def trace_from_iface_dyn(self,
                             pop,
                             iface_low,
                             iface_high,
                             timeout=100.0,
                             speriod=0.01,
                             min_samples=5):
        """
        Returns a pair of (res, time), res being 1 (up), -1 (down), 0 (timeout).
        """

        pop.sampled += 1
        state = pop.state.copy()
        state.seed += np.random.randint(1 << 32)  # TODO: deterministic. Modify state?
        t = 0.0  # Time
        samples = 0

        while True:
            updates = max(1, int(speriod * state.n))
            dt = updates / state.n
            state.mc_sweep(sweeps=0, updates=updates)
            t += dt
            self.ran_updates += dt

            cstats = state.mc_max_cluster(samples=self.cluster_samples,
                                          edge_prob=self.cluster_e_prob)
            param = cstats.v_in
            self.ran_clusters += self.cluster_samples
            samples += 1

            if param >= iface_high.param:
                npop = PopSample(param, iface_high.number, pop, t, state, cstats)
                if samples >= min_samples:
                    pop.up_times.append(t)
                    iface_high.pops.append(npop)
                return (1, t, samples, npop)

            if iface_low and param < iface_low.param:
                if samples >= min_samples:
                    pop.down_times.append(t)
                return (-1, t, samples, None)

            if t > timeout:
                if samples >= min_samples:
                    pop.timeouts.append(t)
                return (0, t, samples, None)
