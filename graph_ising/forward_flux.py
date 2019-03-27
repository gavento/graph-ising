import attr
import numpy as np
import tensorflow.compat.v2 as tf

from .ising import GraphSetIsing


@attr.s
class PopSample:
    param = attr.ib(type=float)
    interface = attr.ib(type=int)
    parent = attr.ib(type='PopSample', repr=False)
    time = attr.ib(type=float)
    data = attr.ib(type=np.ndarray, repr=False)

    sampled = attr.ib(0)
    up_count = attr.ib(0)
    up_times = attr.ib(factory=list)
    down_count = attr.ib(0)
    down_times = attr.ib(factory=list)


class DirectIsingClusterFFSampler:

    def __init__(self, graph, interfaces, batch_size=100, pop_size=100, init_spins=-1.0, **kwargs):
        self.graph = graph
        self.ising = GraphSetIsing(graphs=[graph] * batch_size, **kwargs)
        self.pop_size = pop_size
        self.batch_size = batch_size

        assert interfaces[0] == 0
        self.interfaces = interfaces
        self.pops = [[] for b in self.interfaces]
        self.pops[0] = [PopSample(0.0, 0, None, 0.0, np.full([graph.order()], init_spins))]

    def sample_batch(self, interface_no):
        """
        Sample a batch of PopSample, returning `(pop_samples, concat_data)`.
        """
        isamples = self.pops[interface_no]
        assert len(isamples) > 0
        idxs = np.random.randint(0, len(isamples), [self.batch_size])
        pops = [isamples[r] for r in idxs]
        return (pops, np.concatenate([p.data for p in pops]))

    def run_batch_from(self, interface_no, steps, update_fraction, clusters_every=100):
        batch_pops, batch_data = self.sample_batch(interface_no)
        step_data = tf.Variable(
            np.zeros((steps + 1,) + tuple(batch_data.shape), dtype=self.ising.ftype),
            trainable=False,
            name='step_data')
        step_data[0].assign(batch_data)
        n_params = steps // clusters_every
        step_params = tf.Variable(
            np.zeros((n_params, self.batch_size), dtype=self.ising.ftype),
            trainable=False,
            name='step_params')
        up = self.interfaces[interface_no + 1] if interface_no + 1 < len(self.interfaces) else 1e100
        down = self.interfaces[interface_no - 1] if interface_no > 0 else -1e100
        s = self._run_updates_fn(
            step_data,
            step_params,
            tf.identity(steps),
            tf.identity(update_fraction),
            clusters_every=tf.identity(clusters_every),
            up=up, down=down,
            stop_fraction=0.5)
        step_params = step_params.numpy()
        step_data = step_data.numpy()
        for gi in range(self.ising.n):
            pop = batch_pops[gi]
            pop.sampled += 1
            for pi in range(n_params):
                t = (pi + 1) * clusters_every
                n = self.graph.order()
                spin = step_data[t, gi * n:(gi + 1) * n]
                if step_params[gi, pi] >= up and interface_no + 1 < len(self.interfaces):
                    npop = PopSample(step_params[gi, pi], interface_no + 1, parent=pop, time=t, data=spin)
                    pop.up_count += 1
                    pop.up_times.append(t)
                    self.pops[interface_no + 1].append(npop)
                if step_params[gi, pi] <= down and interface_no > 0:
                    npop = PopSample(step_params[gi, pi], interface_no - 1, parent=pop, time=t, data=spin)
                    pop.down_count += 1
                    pop.down_times.append(t)
                    self.pops[interface_no - 1].append(npop)

    def select_spins_in_times(self, step_data, graph_steps):
        """
        Return spin vector selected from step_data, every graph at the given step.
        """
        vertex_steps = tf.gather(graph_steps, self.ising.v_batch)
        return tf.gather(step_data, vertex_steps)

    # @tf.function(input_signature=([
    #     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    #     tf.TensorSpec(shape=[], dtype=tf.int64),
    #     tf.TensorSpec(shape=[], dtype=tf.float32),
    #     tf.TensorSpec(shape=[], dtype=tf.int64),
    #     tf.TensorSpec(shape=[], dtype=tf.float32),
    #     tf.TensorSpec(shape=[], dtype=tf.float32),
    #     ]))
    @tf.function
    def _run_updates_fn(self, step_data, step_params, steps, update_fraction, clusters_every, up,
                        down, stop_fraction):
        for s in range(1, steps + 1):
            d2 = self.ising.update(step_data[s - 1], update_fraction)
            step_data[s].assign(d2)
            if (s > 0) & tf.equal(s % clusters_every, 0):
                p2 = self.ising.largest_clusters(d2)
                step_params[(s // clusters_every) - 1].assign(p2)
                # TODO: Stop early if up/down params are hit for stop_fraction
                done = tf.reduce_sum(tf.cast((p2 >= up) | (p2 <= down), tf.float32))
                #tf.print(done > (stop_fraction * tf.cast(self.ising.v_n, tf.float32)))
                #if done > (stop_fraction * tf.cast(self.ising.v_n, tf.float32)):
                #    pass
                #    return s
        #return steps
