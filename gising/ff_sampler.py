import time
import tqdm
import attr
import numpy as np
import tensorflow.compat.v2 as tf
import types
from .cising import IsingState, ClusterStats


def stat_str(xs, minmax=False, prec=3):
    if isinstance(xs, types.GeneratorType):
        xs = np.array(xs)
    if len(xs) == 0:
        return "[0x]"
    s = f"[{len(xs)}x {np.mean(xs):.{prec}g}±{np.std(xs):.{prec}g}]"
    if minmax:
        s = s[:-1] + f", {np.min(xs):.{prec}g} .. {np.max(xs):.{prec}g}]"
    return s


@attr.s
class PopSample:
    param = attr.ib(type=float)
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

    def get_time_estimate(self, quantile=0.2, base_estimate=0.1):
        Ts = []
        for p in self.pops:
            Ts.extend(p.up_times)
            Ts.extend(p.down_times)
        if Ts:
            return np.quantile(Ts, quantile)
        return base_estimate

    def normalized_upflow(self, width):
        up = len(self.up_times())
        oth = len(self.down_times()) + len(self.timeouts())
        return (up / (up + oth))**(1 / width)

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
        self.cross_A_samples = 10 * min_pop_size
        self.cluster_samples = cluster_samples
        self.cluster_e_prob = cluster_e_prob

        self.ran_updates = 0.0
        self.ran_clusters = 0

        self.interfaces = [
            iface if isinstance(iface, Interface) else Interface(iface) for iface in interfaces
        ]
        self.start_pop = None
        self.ifaceA = self.interfaces[0]
        self.ifaceB = self.interfaces[-1]

    def fill_interfaces(self, progress=False, timeout=100.0, tgt_samples=10, time_est=0.1):
        min_samples = tgt_samples / 2
        # Sample interface A
        up_up_times, npops = self.sample_up_crosses(self.start_pop,
                                                    self.ifaceA.param,
                                                    self.cross_A_samples,
                                                    progress=progress,
                                                    timeout=timeout)
        self.ifaceA.pops = npops
        self.ifaceA_up_up_times = up_up_times

        for ino, iface in enumerate(self.interfaces):
            if ino == 0:
                continue
            bot = self.ifaceA
            prev = self.interfaces[ino - 1]
            if progress:
                pb = tqdm.tqdm(range(self.min_pop_size),
                               f"Gen interface {ino} [{iface.param:6.2f}]")
            while len(iface.pops) < self.min_pop_size:
                time_est = prev.get_time_estimate(base_estimate=time_est)
                pop = prev.get_random_pop()
                speriod = time_est / tgt_samples
                self.trace_pop(pop,
                               bot,
                               iface,
                               timeout=timeout,
                               speriod=speriod,
                               min_samples=min_samples,
                               max_over_param=0.5)
                if progress:
                    pb.update(len(iface.pops) - pb.n)
                    pb.set_postfix_str(
                        f"{len(prev.up_times())}U {len(prev.down_times())}D {len(prev.timeouts())}TO {speriod:.3g} swp/s"
                    )
                    pb.display()
            if progress:
                pb.display()
                pb.close()
            print(
                f"  genrated {ino} / {len(self.interfaces)}" +
                f", Param {stat_str([p.param for p in iface.pops], True)} tgt {iface.param:.3g}" +
                (f", UpTime {stat_str(prev.up_times(), True)}, DownTime {stat_str(prev.down_times(), True)}"
                ))


class CIsingFFSampler(FFSampler):

    def __init__(self, graph, interfaces, state=None, **kwargs):
        super().__init__(graph, interfaces, **kwargs)
        if state is None:
            state = IsingState(graph=graph)
        self.start_pop = PopSample(0.0, None, 0.0, state, None)

    def trace_pop(self,
                  pop,
                  iface_low,
                  iface_high,
                  timeout=100.0,
                  speriod=0.01,
                  min_samples=5,
                  max_over_param=0.5):
        """
        Returns None.
        """

        pop.sampled += 1
        state = pop.state.copy()
        state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?
        t = 0.0  # Time
        samples = 0  # Clusterings done

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
                npop = PopSample(param, pop, t, state, cstats)
                pop.up_times.append(t)
                if samples >= min_samples and param < iface_high.param + max_over_param:
                    iface_high.pops.append(npop)
                return

            if iface_low and param < iface_low.param:
                pop.down_times.append(t)
                return

            if t > timeout:
                pop.timeouts.append(t)
                return

    def sample_up_crosses(self,
                          pop0,
                          threshold,
                          target,
                          timeout=100.0,
                          speriod=0.01,
                          progress=False):
        """
        Returns (up_to_up_times, pops)
        """
        updates = max(1, int(speriod * pop0.state.n))
        dt = updates / pop0.state.n
        up_to_up_times = []
        npops = []

        state = pop0.state.copy()
        state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?
        t = 0.0
        up = pop0.param >= threshold
        t_up = None
        if progress:
            r = tqdm.tqdm(range(target), f"Computing up-cross rate at {threshold:.3g}")
        while True:
            state.mc_sweep(sweeps=0, updates=updates)
            t += dt
            self.ran_updates += dt
            cstats = state.mc_max_cluster(samples=self.cluster_samples,
                                          edge_prob=self.cluster_e_prob)
            self.ran_clusters += self.cluster_samples
            param = cstats.v_in

            if (t - (t_up or 0.0) > timeout):  # TODO: periodic resets to re-seed?
                # Reset to pop0
                t = 0.0
                up = pop0.param >= threshold
                t_up = None
                state = pop0.state.copy()
                state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?

            if param >= threshold:
                if not up:
                    if t_up is not None:
                        up_to_up_times.append(t - t_up)
                        if param < threshold + 0.5:
                            npops.append(PopSample(param, pop0, t, state.copy(), cstats))
                            if progress:
                                r.update(1)
                    t_up = t
                    if progress:
                        r.set_postfix_str(f"up-up sweeps {stat_str(up_to_up_times, True)}")
                        r.display()
                    up = True
            else:
                up = False

            if len(npops) >= target:
                return (up_to_up_times, npops)
