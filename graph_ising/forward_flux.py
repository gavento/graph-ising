import attr
import numpy as np
import tensorflow.compat.v2 as tf

from .ising import GraphSetIsing


@attr.s
class PopSample:
    param = attr.ib(type=float)
    interface = attr.ib(type='Interface')
    parent = attr.ib(type='PopSample', repr=False)
    time = attr.ib(type=float)
    data = attr.ib(type=np.ndarray, repr=False)
    E = attr.ib(0.0, type=float)

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
        return np.array([p.E for p in self.pops])

    def down_times(self):
        if not self.pops:
            return np.zeros(0)
        return np.concatenate([p.down_times for p in self.pops])

    def timeouts(self):
        if not self.pops:
            return np.zeros(0)
        return np.concatenate([p.timeouts for p in self.pops])

    def sample_batch(self, batch_size):
        """
        Sample a batch of PopSample, returning `(pops, data_tensor)`.
        """
        assert len(self.pops) > 0
        idxs = np.random.randint(0, len(self.pops), [batch_size])
        pops = [self.pops[r] for r in idxs]
        return (pops, np.concatenate([p.data for p in pops]))

    def sim_time(self, *, min_samples=3, base_time=10.0):
        """
        Return a time suitable for the next run.
        
        Max of:
          2 * (90% percentile of down-times) if any and meaningful,
          2 * (90% percentile of up-times) if any and meaningful,
          2 * max time sampled if any of the above are meaningful but have <min_samples entries,
          or base_time if none above apply.
        """
        vs = []
        ups = self.up_times()
        downs = self.down_times()
        tos = self.timeouts()
        allts = np.concatenate((ups, downs, tos))

        if len(ups) > 0:
            vs.append(2 * np.quantile(ups, 0.9))
        if len(downs) > 0:
            vs.append(2 * np.quantile(downs, 0.9))
        if len(ups) < min_samples or len(downs) < min_samples:
            if len(allts) > 0:
                vs.append(2 * np.max(allts))
        if vs:
            return max(vs)
        else:
            return base_time

    def __repr__(self):
        uts = self.up_times()
        dts = self.down_times()
        mut = np.mean(uts) if len(uts) > 0 else 0.0
        mdt = np.mean(dts) if len(dts) > 0 else 0.0
        tot = len(uts) + len(dts)
        frac = len(uts) / tot if tot > 0 else 0.0
        return "{}(param={}, {} pops, {} ups (time {:.2g}), {} downs (time {:.2g})), {:.4f} up".format(
            self.__class__.__name__, self.param, len(self.pops), len(uts), mut, len(dts), mdt, frac)


class DirectIsingClusterFFSampler:

    def __init__(self,
                 graph,
                 interfaces,
                 batch_size=100,
                 pop_size=100,
                 update_fraction=0.1,
                 init_spins=-1.0,
                 drop_edges=0.0,
                 drop_samples=1,
                 **kwargs):
        self.graph = graph
        self.ising = GraphSetIsing(graphs=[graph] * batch_size, **kwargs)
        self.pop_size = pop_size
        self.batch_size = batch_size
        self.update_fraction = update_fraction
        self.drop_edges = drop_edges
        self.drop_samples = drop_samples

        self.interfaces = [i if isinstance(i, Interface) else Interface(i) for i in interfaces]
        assert self.interfaces[0].param == 0.0
        self.interfaces[0].pops.append(
            PopSample(0.0, 0, None, 0.0, np.full([graph.order()], init_spins)))

    def run_adaptive_batch(self, interface, cluster_times=20):
        assert interface.pops
        steps = max(int(interface.sim_time() / self.update_fraction), cluster_times)
        self.run_batch(interface, steps, steps // cluster_times)

    def run_batch(self, interface, steps, clusters_every=100):
        if isinstance(interface, int):
            interface_no = interface
            interface = self.interfaces[interface_no]
        else:
            assert isinstance(interface, Interface)
            interface_no = self.interfaces.index(interface)

        batch_pops, batch_data = interface.sample_batch(self.batch_size)
        #step_data = tf.Variable(
        #    np.zeros((steps + 1,) + tuple(batch_data.shape), dtype=self.ising.ftype),
        #    trainable=False,
        #    name='step_data')
        #step_data[0].assign(batch_data)
        n_params = steps // clusters_every
        #step_params = tf.Variable(
        #    np.zeros((n_params, self.batch_size), dtype=self.ising.ftype),
        #    trainable=False,
        #    name='step_params')
        up = self.interfaces[interface_no +
                             1].param if interface_no + 1 < len(self.interfaces) else 1e100
        down = self.interfaces[interface_no - 1].param if interface_no > 0 else -1e100
        step_data, step_params = self._run_updates_fn(
            batch_data,
            #step_data,
            #step_params,
            tf.identity(steps),
            tf.identity(self.update_fraction),
            clusters_every=tf.identity(clusters_every),
            #up=up, down=down
        )
        step_params = step_params.numpy()
        step_data = step_data.numpy()
        for gi in range(self.ising.n):
            pop = batch_pops[gi]
            pop.sampled += 1
            for pi in range(n_params):
                s = (pi + 1) * clusters_every
                t = s * self.update_fraction
                n = self.graph.order()
                spin = step_data[pi, gi * n:(gi + 1) * n]
                if step_params[pi, gi] >= up and interface_no + 1 < len(self.interfaces):
                    npop = PopSample(
                        step_params[pi, gi], interface_no + 1, parent=pop, time=t, data=spin)
                    pop.up_times.append(t)
                    self.interfaces[interface_no + 1].pops.append(npop)
                    break
                if step_params[pi, gi] <= down and interface_no > 0:
                    npop = PopSample(
                        step_params[pi, gi], interface_no - 1, parent=pop, time=t, data=spin)
                    pop.down_times.append(t)
                    self.interfaces[interface_no - 1].pops.append(npop)
                    break
            else:
                # No interface was hit
                pop.timeouts.append(steps * self.update_fraction)

    def select_spins_in_times(self, step_data, graph_steps):
        """
        Return spin vector selected from step_data, every graph at the given step.
        """
        vertex_steps = tf.gather(graph_steps, self.ising.v_batch)
        return tf.gather(step_data, vertex_steps)

    @tf.function(
        input_signature=([
            #tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            #tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ]))
    #@tf.function
    def _run_updates_fn(self, batch_data, steps, update_fraction, clusters_every):
        ds = tf.TensorArray(tf.float32, steps + 1, clear_after_read=False)
        ps = tf.TensorArray(tf.float32, steps // clusters_every)
        for s in range(1, steps + 1):
            #tf.print(s)
            #tf.print(batch_data)
            d2 = self.ising.update(batch_data, update_fraction)
            if tf.equal(s % clusters_every, 0):
                p2 = self.ising.largest_clusters(d2, samples=tf.constant(self.drop_samples), drop_edges=tf.constant(self.drop_edges, tf.float32))
                idx = (s // clusters_every) - 1
                ds = ds.write(idx, d2)
                ps = ps.write(idx, p2)
                #tf.print(p2)
                # NO-DO: Stop early if up/down params are hit for stop_fraction
                #done = tf.reduce_sum(tf.cast((p2 >= up) | (p2 <= down), tf.float32))
                #tf.print(done > (stop_fraction * tf.cast(self.ising.v_n, tf.float32)))
                #if done > (stop_fraction * tf.cast(self.ising.v_n, tf.float32)):
                #    pass
                #    return s
            batch_data = d2
        #tf.print(ds)
        return ds.stack(), ps.stack()
