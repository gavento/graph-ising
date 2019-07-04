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
    log10_rate = attr.ib(0.0)
    s_up = attr.ib(0)
    s_down = attr.ib(0)
    s_timeout = attr.ib(0)

    def up_flow(self):
        return (self.s_up / max(self.s_up + self.s_down, 1))

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.order:.4g}) {len(self.states)} samples, {self.s_up}U {self.s_down}D {self.s_timeout}TO, rate {self.rate:.3g}>"

    def reset_counts(self):
        self.s_up = 0
        self.s_down = 0
        self.s_timeout = 0


class FFSampler:

    def __init__(self, init_states, interfaces, iface_samples=100):
        self.init_states = init_states
        self.iface_samples = iface_samples

        self.interfaces = [
            iface if isinstance(iface, FFInterface) else FFInterface(iface) for iface in interfaces
        ]
        self.ifaceA = self.interfaces[0]
        self.ifaceB = self.interfaces[-1]

    def compute(self, progress=True, report_degs=False, timeout=100.0, dynamic_ifaces=False, stop_rate=None):
        self.sample_interface_A(progress=progress, timeout=timeout)
        print(f"Rate at iface A ({self.ifaceA.order}) is {10 ** self.ifaceA.log10_rate:.3g} ups/MCSS/spin")
        step = 10
        maxstep = max((self.ifaceB.order - self.ifaceA.order) // 20, 1)
        ino = 1

        while True:
            prev = self.interfaces[ino - 1]
            if not dynamic_ifaces:
                iface = self.interfaces[ino]
            else:
                its = 0
                last_dir = 0
                while True:
                    iface = FFInterface(min(prev.order + step, self.ifaceB.order))
                    ok = self.sample_interface(iface, prev=prev, progress=False, timeout=timeout, iface_samples=10, max_timeouts=1)
                    upflow = prev.up_flow()
                    if not ok:
                        print(f"  .. failed to estimate step at {iface.order}, too many timeouts (upflow {upflow:.3f}, step {step})")
                    prev.reset_counts()
                    if False and its > 0:
                        print(f"  .. tried {iface.order} (step {step}), upflow {upflow:.3f}")
                    its += 1

                    if upflow >= 0.5 and step < maxstep and its < 10 and last_dir >= 0:
                        step = min(max(int(step * 2), step + 1), maxstep)
                        last_dir = 1
                        continue
                    elif upflow <= 0.15 and step > 1 and its < 10:
                        step = step * 2 // 3
                        last_dir = -1
                        continue
                    elif iface.order == self.ifaceB.order:
                        iface = self.ifaceB
                        break
                    else:
                        self.interfaces.insert(-1, FFInterface(iface.order))
                        iface = self.interfaces[ino]
                        break
                
            self.sample_interface(iface, prev=prev, progress=progress, timeout=timeout)

            s = f"done {ino}/{len(self.interfaces)} ifaces [{iface.order}]"
            if dynamic_ifaces:
                s = f"done [{iface.order}/{self.ifaceB.order}]"
            up_norm = prev.up_flow() ** (1 / (iface.order - prev.order))
            print(f"  {s}, up flow {prev.up_flow():.3f} (normalized {up_norm:.3f}), rate 10^{iface.log10_rate:.3f}={10**iface.log10_rate:.3g}, " +
                  f"orders {stat_str([s.get_order() for s in iface.states], True)}")

            ino += 1
            if dynamic_ifaces and self.ifaceB == iface:
                break
            if (not dynamic_ifaces) and ino == len(self.interfaces):
                break
            if stop_rate is not None and stop_rate > iface.log10_rate:
                print(f"  Rate below stop_rate 10^{stop_rate:.3f}, stopping")
                break

    def sample_interface_A(self, progress, timeout):
        up_times = []
        a = self.ifaceA.order

        if progress:
            pb = tqdm.tqdm(range(self.iface_samples),
                           f"Iface A ({a:.2f}) rate",
                           dynamic_ncols=True,
                           leave=False,
                           file=progress if progress is not True else sys.stderr)

        state = None
        t_up = None
        timeouts = 0

        while min(len(up_times), len(self.ifaceA.states)) < self.iface_samples:
            if progress:
                pb.set_postfix_str(f"times {stat_str(up_times, True)}, {timeouts} TOs")
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

        self.ifaceA.log10_rate = np.log10(1.0 / np.mean(up_times))

        if progress:
            pb.update(min(len(up_times), len(self.ifaceA.states)) - pb.n)
            pb.close()
            print(pb)

    def sample_interface(self, iface, prev, progress, timeout, iface_samples=None, max_timeouts=None):
        "Return False on too many timeouts"
        if iface_samples is None:
            iface_samples = self.iface_samples
        if progress:
            pb = tqdm.tqdm(range(iface_samples),
                           f"Iface {iface.order:8.2f}",
                           dynamic_ncols=True,
                           leave=False,
                           file=progress if progress is not True else sys.stderr)

        while len(iface.states) < iface_samples:
            # Select clustering seed for this pop
            state = np.random.choice(prev.states).copy()
            state.seed = np.random.randint(1 << 60)
            state.update_until(self.ifaceA.order, iface.order, timeout=timeout)
            if state.get_order() < self.ifaceA.order:
                prev.s_down += 1
            elif state.get_order() >= iface.order:
                prev.s_up += 1
                iface.states.append(state.copy())
            else:
                prev.s_timeout += 1
                if max_timeouts is not None and prev.s_timeout >= max_timeouts:
                    return False

            if progress:
                pb.update(len(iface.states) - pb.n)
                pb.set_postfix_str(f"{prev.s_up:>3}U {prev.s_down:>3}D {prev.s_timeout:>3}TO")

        if progress:
            pb.update(len(iface.states) - pb.n)
            pb.close()
            print(pb)

        iface.log10_rate = prev.log10_rate + np.log10(prev.up_flow())
        return True

    def critical_order_param(self):
        last_r = self.ifaceB.log10_rate
        if last_r == 0.0:
            return None
        for ino, iface in enumerate(self.interfaces):
            if iface.log10_rate < last_r + np.log10(2.0):
                break
        if ino == 0:
            return 0.0
        prev = self.interfaces[ino - 1]
        # print(f"Locating {last_r * 2.0} in {prev.rate} .. {iface.rate} ({prev.order} .. {iface.order})")
        la = prev.log10_rate
        lx = last_r  + np.log10(2.0)
        lb = iface.log10_rate
        return ((lx - la) * iface.order + (lb - lx) * prev.order) / (lb - la)
