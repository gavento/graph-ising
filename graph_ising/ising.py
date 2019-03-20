import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf

from .graph_set import GraphSet


class GraphSetIsing(GraphSet):

    _VARS = GraphSet._VARS + [
        ('T', [None], np.float32),
        ('F', [None], np.float32),
        ('J', [None], np.float32),
        ('p_update', [None], np.float32),
        ]

    def __init__(self, graphs, T=1.0, F=0.0, J=1.0, p_update=1.0):
        super().__init__(graphs)
        def expand_to_every(v, sizes, dtype=np.float32):
            if isinstance(v, (int, float, np.number)):
                v = [v] * len(sizes)
            r = []
            for x, s in zip(v, sizes):
                if isinstance(x, (int, float, np.number)):
                    r.extend([x] * s)
                else:
                    assert(len(x) == s)
                    r.extend(x)
            return np.array(r, dtype=dtype)
        self.T = expand_to_every(T, self.orders)
        self.F = expand_to_every(F, self.orders)
        self.J = expand_to_every(J, self.sizes)
        self.p_update = expand_to_every(p_update, self.orders)

        self._add_copy_var('T')
        self._add_copy_var('F')
        self._add_copy_var('J')
        self._add_copy_var('p_update')

        self.m_update_flipped = self._add_mean_metric('update/flipped')
        self.m_update_mean_spin = self._add_mean_metric('update/mean_spin')
        self.m_components_unfinished = self._add_mean_metric('components/unfinished')
        self.m_components_iterations = self._add_mean_metric('components/iterations')


    def update(self, spins_in, update_fraction=tf.constant(1.0), update_metrics=True):
        "Returns a resulting spins_out tensor operation"
        sum_neighbors = self.sum_neighbors(spins_in, edge_weights=self.v_J)
        # delta energy if flipped
        delta_E = (self.v_F + sum_neighbors) * 2.0 * spins_in
        # probability of random flip
        random_flip = tf.random.uniform((self.order, ), name='flip_p') < tf.math.exp(-delta_E / self.v_T)
        # updating only a random subset
        update = (tf.random.uniform((self.order, ), name='update_p') < self.v_p_update * update_fraction)
        # combined condition
        flip = ((delta_E < 0.0) | random_flip) & update
        # update spins
        spins_out = (1.0 - tf.cast(flip, tf.float32) * 2.0) * spins_in
        # metrics
        self.metric_update_flipped.update_state(tf.cast(flip, tf.float32), update) # TODO: distinguish with self.v_p_update
        self.metric_update_mean_spin.update_state(spins_out)
        return spins_out
