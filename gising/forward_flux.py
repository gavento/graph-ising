import sys
import time

import attr
import numpy as np
import tqdm

from .utils import stat_str


class FFSampler:
    pass


@attr.s
class FFInterface:
    order = attr.ib(type=float)
    states = attr.ib(factory=list)
    rate = attr.ib(0.0)
    s_up = attr.ib(0)
    s_down = attr.ib(0)
    s_timeout = attr.ib(0)

    def up_flow(self):
        return (self.s_up / max(self.s_up + self.s_down, 1))

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.order:.4g}) {len(self.states)} samples, {self.s_up}U {self.s_down}D {self.s_timeout}TO, rate {self.rate:.3g}>"


class FFSampler:

    def __init__(self, init_states, interfaces, iface_samples=100):
        self.init_states = init_states
        self.iface_samples = iface_samples

        self.interfaces = [
            iface if isinstance(iface, FFInterface) else FFInterface(iface) for iface in interfaces
        ]
        self.ifaceA = self.interfaces[0]

    def compute(self, progress=True, timeout=100.0):
        self.sample_interface_A(progress=progress, timeout=timeout)
        print(f"Rate at iface A ({self.ifaceA.order}) is {self.ifaceA.rate:.3g} ups/MCSS/spin")

        for ino, iface in enumerate(self.interfaces):
            if ino == 0:
                continue
            self.sample_interface(iface,
                                  prev=self.interfaces[ino - 1],
                                  progress=progress,
                                  timeout=timeout)

            print(f"  done {ino}/{len(self.interfaces)}, rate {iface.rate:.3g}, " +
                  f"orders {stat_str([s.get_order() for s in iface.states], True)}")

            # Report in-cluster degrees and other stats
            s = iface.states[0]
            mask = s.get_stats().mask

            dgs = [0] * s.graph.size()
            dgc = [0] * s.graph.size()
            for v in range(s.n):
                d = s.graph.degree(v)
                dgs[d] += 1
                if mask[v] > 0:
                    dgc[d] += 1
            dgstr = ' '.join(f"{d}:{c}/{g}" for d, (g, c) in enumerate(zip(dgs, dgc)) if g > 0)
            print(f"  one cluster degs: {dgstr}")

    def sample_interface_A(self, progress, timeout):
        up_times = []
        a = self.ifaceA.order

        if progress:
            pb = tqdm.tqdm(range(self.iface_samples),
                           f"Iface A ({a:.3g}) rate",
                           dynamic_ncols=True,
                           leave=False,
                           file=progress if progress is not True else sys.stderr)

        state = None
        t_up = None
        timeouts = 0

        while min(len(up_times), len(self.ifaceA.states)) < self.iface_samples:
            if progress:
                pb.set_postfix_str(f"times {stat_str(up_times, True)}, {timeouts} timeouts")
                pb.display()

            if state is None:
                t_up = None
                state = np.random.choice(self.init_states).copy()
                state.seed = np.random.randint(1 << 60)

            # Update to be <A
            state.update_until(a, 1 << 30, timeout=timeout)
            if state.get_order() >= a:
                state = None
                timeouts += 1
                if t_up is not None:
                    up_times.append(timeout)
                continue

            # Update to be >=A
            state.update_until(0, self.ifaceA.order, timeout=timeout)
            if state.get_order() < a:
                state = None
                timeouts += 1
                if t_up is not None:
                    up_times.append(timeout)
                continue

            if t_up is not None:
                up_times.append(state.updates - t_up)
            t_up = state.updates

            self.ifaceA.states.append(state.copy())

            if progress:
                pb.update(min(len(up_times), len(self.ifaceA.states)) - pb.n)

        self.ifaceA.rate = 1.0 / np.mean(up_times)

        if progress:
            pb.update(min(len(up_times), len(self.ifaceA.states)) - pb.n)
            pb.close()
            print(pb)

    def sample_interface(self, iface, prev, progress, timeout):
        if progress:
            pb = tqdm.tqdm(range(self.iface_samples),
                           f"Iface {iface.order:8.3g}",
                           dynamic_ncols=True,
                           leave=False,
                           file=progress if progress is not True else sys.stderr)

        while len(iface.states) < self.iface_samples:
            if progress:
                pb.set_postfix_str(f"{prev.s_up:>3}U {prev.s_down:>3}D {prev.s_timeout:>3}TO")
                pb.update(len(iface.states) - pb.n)

            # Select clustering seed for this pop
            state = np.random.choice(prev.states).copy()
            state.update_until(self.ifaceA.order, iface.order, timeout=timeout)
            if state.get_order() < self.ifaceA.order:
                prev.s_down += 1
            elif state.get_order() >= iface.order:
                prev.s_up += 1
                iface.states.append(state.copy())
            else:
                prev.s_timeout += 1

        if progress:
            pb.update(len(iface.states) - pb.n)
            pb.close()
            print(pb)

        iface.rate = prev.rate * prev.up_flow()
