import sys
import time

import attr
import numpy as np
import tensorflow.compat.v2 as tf
import tqdm

from .cising import ClusterStats, IsingState
from .utils import stat_str


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
    rate = attr.ib(0.0, type=float)

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

    def get_random_pop(self, param_below=None, cseed=None, **kwargs):
        assert len(self.pops) > 0
        for i in range(1000):
            p = self.pops[np.random.randint(0, len(self.pops))]  # TODO: Use the seed?
            param = p.param
            if cseed is not None and param_below is not None:
                cstats = p.state.mc_max_cluster(seed=cseed, **kwargs)
                param = cstats.v_in
            if param_below is None or param < param_below:
                return p
        raise Exception("Unable to sample pop within param range")

    def get_time_estimate(self, quantile=0.1, base_estimate=0.1):
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
        if up == 0:
            oth = 1.0
        return (up / (up + oth))**(1.0 / width)

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
        self.cross_A_samples = 2 * min_pop_size
        self.cluster_samples = cluster_samples
        self.cluster_e_prob = cluster_e_prob

        self.interfaces = [
            iface if isinstance(iface, Interface) else Interface(iface) for iface in interfaces
        ]
        self.start_pop = None
        self.ifaceA = self.interfaces[0]
        self.ifaceB = self.interfaces[-1]

    def fill_interfaces(self, progress=False, timeout=100.0, tgt_samples=10, time_est=0.1):
        # Sample interface A
        up_up_times, npops = self.sample_up_crosses(self.start_pop,
                                                    self.ifaceA.param,
                                                    self.cross_A_samples,
                                                    progress=progress,
                                                    timeout=timeout)
        self.ifaceA.pops = npops
        self.ifaceA.rate = 1.0 / np.mean(up_up_times) / self.graph.order()

        for ino, iface in enumerate(self.interfaces):
            if ino == 0:
                continue
            bot = self.ifaceA
            prev = self.interfaces[ino - 1]
            if progress:
                pb = tqdm.tqdm(range(self.min_pop_size),
                               f"Gen interface {ino} [{iface.param:6.2f}]",
                               dynamic_ncols=True,
                               leave=False,
                               file=progress if progress is not True else sys.stderr)
            while len(iface.pops) < self.min_pop_size:
                # Select clustering seed for this pop
                cseed = np.random.randint(1 << 60)
                pop = prev.get_random_pop(iface.param,
                                          cseed=cseed,
                                          samples=self.cluster_samples,
                                          edge_prob=self.cluster_e_prob)
                speriod = 0.05  # min(max(time_est / tgt_samples, 0.01), 0.1)
                self.trace_pop(pop, bot, iface, timeout=timeout, speriod=speriod, cseed=cseed)
                if progress:
                    pb.update(len(iface.pops) - pb.n)
                    pb.set_postfix_str(
                        f"{len(prev.up_times())}U {len(prev.down_times())}D {len(prev.timeouts())}TO {speriod:.3g} swp/s"
                    )
                    pb.display()
            if progress:
                pb.display()
                pb.close()
                print(pb)
                del pb
            iface.rate = prev.rate * prev.normalized_upflow(1.0)
            sys.stderr.write(
                f"  done {ino} [{iface.param:.3g}] of {len(self.interfaces)}, rate {iface.rate:.3g}"
                + f", Param {stat_str([p.param for p in iface.pops], True)}" +
                f", UpTime {stat_str(prev.up_times(), True)}, DownTime {stat_str(prev.down_times(), True)}\n"
            )

            # Report in-cluster degrees and other stats
            cs = iface.pops[0].cluster_stats
            dgs = [0] * self.graph.size()
            dgc = [0] * self.graph.size()
            for v in range(self.graph.order()):
                d = self.graph.degree(v)
                dgs[d] += 1
                if cs.mask[v] > 0:
                    dgc[d] += 1
            dgstr = ' '.join(f"{d}:{c}/{g}" for d, (g, c) in enumerate(zip(dgs, dgc)) if g > 0)
            sys.stderr.write(
                f"  one cluster: V={cs.v_in} E={cs.e_in} Eout={cs.e_border} degs: {dgstr}\n"
            )


class CIsingFFSampler(FFSampler):

    def __init__(self, graph, interfaces, state=None, **kwargs):
        super().__init__(graph, interfaces, **kwargs)
        if state is None:
            state = IsingState(graph=graph)
        self.start_pop = PopSample(0.0, None, 0.0, state, None)

    def run_sweep_up(self, s0, up, sweeps=0.1, up_accuracy=0.1, cseed=None):
        """
        Runs sim for `sweeps` and  
        Assumes that state param is strictly below `up`.
    
        Returns (final_state, cluster_stats).
        """
        state = s0.copy()

        ### Assert
        cstats0 = state.mc_max_cluster(samples=self.cluster_samples,
                                       edge_prob=self.cluster_e_prob,
                                       seed=cseed)
        if cstats0.v_in >= up:
            print(f"  - run_sweep_up param {cstats0.v_in} not below up ({up}) at {s0}")

        updates = max(1, int(sweeps * state.n))
        state.mc_sweep(sweeps=0, updates=updates)
        cstats = state.mc_max_cluster(samples=self.cluster_samples,
                                      edge_prob=self.cluster_e_prob,
                                      seed=cseed)
        param = cstats.v_in
        if param <= up + up_accuracy:
            # Success
            return (state, cstats)

        # Param above up + up_accuracy -> do bisection
        updates_hi = updates
        updates_lo = 0
        runs = 0
        while True:
            state = s0.copy()
            updates = (updates_hi + updates_lo + 1) // 2

            state.mc_sweep(sweeps=0, updates=updates)
            cstats = state.mc_max_cluster(samples=self.cluster_samples,
                                          edge_prob=self.cluster_e_prob,
                                          seed=cseed)
            param = cstats.v_in

            if (param >= up and param <= up + up_accuracy) or (updates == updates_hi):
                # Success or minimal step
                return (state, cstats)
            if param < up:
                updates_lo = updates
            if param > up + up_accuracy:
                updates_hi = updates

            runs += 1
            assert runs < 250  # Bisection halving should never get here

    def trace_pop(self,
                  pop,
                  iface_low,
                  iface_high,
                  timeout=100.0,
                  speriod=0.1,
                  min_samples=5,
                  max_over_param=0.5,
                  cseed=None):
        """
        Returns None.
        """
        pop.sampled += 1
        state = pop.state.copy()
        state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?
        t0 = state.time  # Time

        while True:
            state, cstats = self.run_sweep_up(state, iface_high.param, sweeps=speriod, cseed=cseed)
            elapsed = state.time - t0
            param = cstats.v_in

            if param >= iface_high.param:
                npop = PopSample(param, pop, elapsed, state, cstats)
                pop.up_times.append(elapsed)
                iface_high.pops.append(npop)
                return

            if iface_low and param < iface_low.param:
                pop.down_times.append(elapsed)
                return

            if elapsed > timeout:
                pop.timeouts.append(elapsed)
                return

    def sample_up_crosses(self,
                          pop0,
                          threshold,
                          samples,
                          timeout=100.0,
                          speriod=0.05,
                          progress=False):
        """
        Returns (up_to_up_times, pops)
        """
        up_to_up_times = []
        npops = []
        timeouts = 0

        state = pop0.state.copy()
        state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?
        cseed = state.seed
        t0 = state.time
        up = pop0.param >= threshold
        t_up = None

        if progress:
            pb = tqdm.tqdm(range(samples),
                           f"Up-rate at {threshold:.3g}",
                           dynamic_ncols=True,
                           leave=False,
                           file=progress if progress is not True else sys.stderr)

        its = 0
        while True:
            its += 1
            state, cstats = self.run_sweep_up(state,
                                              np.inf if up else threshold,
                                              sweeps=speriod,
                                              cseed=cseed)
            param = cstats.v_in

            if progress and its % 10 == 0:
                pb.set_postfix_str(
                    f"param {param:.3g}, {timeouts} TO, times {stat_str(up_to_up_times, True)}")
                pb.display()

            if (state.time - (t_up or t0) > timeout):  # TODO: periodic resets to re-seed?
                # Reset to pop0
                state = pop0.state.copy()
                state.seed = np.random.randint(1 << 60)  # TODO: Make reproducible? Modify state?
                cseed = state.seed
                t0 = state.time
                up = pop0.param >= threshold
                t_up = None
                timeouts += 1
                up_to_up_times.append(timeout)

            elif param >= threshold:
                if not up:
                    if t_up is not None:
                        up_to_up_times.append(state.time - t_up)
                        npops.append(PopSample(param, pop0, state.time - t_up, state.copy(),
                                               cstats))
                        if progress:
                            pb.update(1)
                    t_up = state.time
                    up = True
            else:  # param < threshold
                up = False

            if len(npops) >= samples:
                if progress:
                    print(pb)
                    pb.close()
                return (up_to_up_times, npops)
