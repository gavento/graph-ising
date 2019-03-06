import contextlib
import time


@contextlib.contextmanager
def timed(name=None):
    t0 = time.time()
    yield
    t1 = time.time()
    print((name + " " if name else "") + "took {:.3f} ms".format((t1 - t0) * 1000.0))
