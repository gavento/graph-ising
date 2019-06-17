try:
    import tensorflow.compat.v2 as tf
except ModuleNotFoundError:
    raise Exception('Tensor API v2 required.')

from .base import Base
from .graph_set import GraphSet
from .ising import GraphSetIsing
from .components import ComponentsMixin
from .utils import timed
