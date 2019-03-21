import contextlib
import time

import tensorflow.compat.v2 as tf

def get_device():
    "Return a valid device, preferably a GPU"
    ds = tf.config.experimental_list_devices()
    for d in ds:
        if 'GPU' in d:
            return d
    return ds[0]


@contextlib.contextmanager
def timed(name=None):
    t0 = time.time()
    yield
    t1 = time.time()
    print((name + " " if name else "") + "took {:.3f} ms".format((t1 - t0) * 1000.0))
