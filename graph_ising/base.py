import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf
import collections


#BaseVar = collections.namedtuple('BaseVar', ['variable', 'orig_name'])


class Base:

    def __init__(self, *, ftype=np.float32, itype=np.int64):
        self.ftype = ftype
        self.itype = itype
        # [Variables by order added]
        self._vars = []
        # [Keras metric object, named]
        self._metrics = []

    def _add_mean_metric(self, name):
        m = tf.keras.metrics.Mean(name)
        self._metrics.append(m)
        return m

    def _add_var(self, name, data, *, prefix='v_', dtype=None):
        var = tf.Variable(data, trainable=False, name=name, dtype=dtype)
        self._vars.append(var)
        self.__setattr__(prefix + name, var)
        return var

    def _add_copy_var(self, name, *, prefix='v_', dtype=None):
        "Simple wrapper, may be replaced by a dynamically slicing mostrosity ;)"
        d = self.__getattribute__(name)
        return self._add_var(name, d, prefix=prefix, dtype=dtype)

    def print_metrics(self, reset=True):
        for m in self._metrics:
            s = "{:20s}: {:.3g}".format(m.name, m.result().numpy())
            print(s)  # Todo: get more fancy
            if reset:
                m.reset_states()

    def write_metrics(self, step, reset=True):
        "Write the metrics to the default summary writer"
        for m in self._metrics:
            tf.summary.scalar(m.name, m.result(), step=step)

    # def _add_copy_var(self, orig_name, *, slice_len=None, capacity=None, prefix='v_', dtype=None):
    #     """
    #     Create a sliced varible proxy for `self.orig_name`.

    #     Updated with `_update_copy_vars`.
    #     The variable is either a direct proxy for `orig_name`, or can accomodate variable 
    #     length data with `slice_len` set to a scalar tensor and `capacity` an upper bound to slice size.
    #     """
    #     d = self.__getattribute__(orig_name, )

    #     if capacity is not None:
    #         raise NotImplementedError("Unfinished, use just the plain version")
    #         assert slice_len is not None
    #         assert isinstance(d, (np.ndarray, tf.Tensor)) and len(d.shape) > 0
    #         shape = list(d.shape)
    #         shape[0] = capacity
    #         var = tf.Variable(tf.zeros(shape, dtype=dtype or d.dtype), trainable=False, name=orig_name, dtype=dtype)
    #         slice_tensor = var[:slice_len]
    #         slice_tensor.assign(d)
    #     else:
    #         assert slice_len is None
    #         var = tf.Variable(d, trainable=False, name=orig_name, dtype=dtype)
    #         slice_tensor = var
    #     bv = BaseVar(var, slice_tensor, slice_len, prefix + orig_name, orig_name)

    #     self.__setattr__(bv.slice_name, bv.slice)
    #     self._vars[orig_name] = bv

    # def _update_var(self, name_or_bv):
    #     if isinstance(name_or_bv, str):
    #         bv = self._vars[name_or_bv]
    #     else:
    #         bv = name_or_bv
    #     bv.slice.assign(self.__getattribute__(bv.orig_name))

    # def _update_copy_vars(self):
    #     "Internal. Update all the TF variables with their current values."
    #     for bv in self._vars.values():
    #         self._update_var(bv)
