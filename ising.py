from cffi import FFI
import numpy as np
import networkx as nx


cising = None
ffi = None

def load_ffi():

    global cising, ffi

    ffi = FFI()
    ffi.cdef("""
        typedef uint64_t rand_t;
        typedef int32_t index_t;
        typedef int8_t spin_t;

        typedef struct {
            index_t n;
            spin_t *spins;
            double field;
            double T;
            rand_t seed;
            index_t *neigh_list; // -1 terminated neighbor list
            index_t *neigh_offset; // for every node starting offset in neigh_list
            index_t *degree; // node degrees
            index_t *degree_sum; // sum of all degrees up to and incl. this node
        } ising_state;

        uint32_t get_rand(rand_t *seed);
        double get_rand_01(rand_t *seed);
        void rand_perm(size_t n, index_t *result, rand_t *seed);

        index_t ising_mc_sweep(ising_state *s);
        index_t ising_max_cluster(ising_state *s, spin_t value, double edge_prob);
        index_t ising_sweep_and_max_cluster(ising_state *s, uint32_t sweeps, spin_t value, double edge_prob);
        """)

    cising = ffi.dlopen('./cising.so')

load_ffi()


class IsingState(object):

    def __init__(self, graph, spins=None, seed=42, field=0.0, T=1.0, defer_init=False):

        self.graph = graph # WARN: NOT A COPY!
        if spins is None:
            self.spins = np.array([1] * self.graph.order(), dtype='int8')
        else:
            self.spins = np.array(spins, dtype='int8')
        self.n = self.spins.shape[0]
        self.field = field
        self.T = T
        self.seed = seed

        assert self.spins.dtype.name == 'int8'
        assert self.spins.shape[0] == self.graph.order()
        assert len(self.spins.shape) == 1
        assert set(self.graph.nodes()) == set(range(len(self.graph.nodes())))

        self.neigh_list = None
        self.neigh_offset = None
        self.degree = None
        self.degree_sum = None

        if not defer_init:
            self.graph_to_internal()

    def copy(self):

        return IsingState(self.spins, self.graph, self.seed, self.field, self.T)

    def graph_to_internal(self):
        "Call this whenever you modify self.graph before a call to any mc_*()"

        self.neigh_list = np.ndarray([2 * self.graph.size() + self.graph.order()], dtype='int32')
        self.neigh_offset = np.ndarray([self.graph.order()], dtype='int32')
        self.degree = np.ndarray([self.graph.order()], dtype='int32')
        self.degree_sum = np.ndarray([self.graph.order()], dtype='int32')

        offset = 0
        for i in sorted(self.graph.nodes()):
            self.neigh_offset[i] = offset
            self.degree[i] = self.graph.degree(i)
            self.degree_sum[i] = self.degree[i] + (0 if i == 0 else self.degree_sum[i - 1])
            for j in self.graph.neighbors(i):
                self.neigh_list[offset] = j
                offset += 1
            self.neigh_list[offset] = -1
            offset += 1

    def internal_to_graph(self):
        "Call this after any graph-modifying mc_* to update self.graph"

        self.graph = nx.empty_graph(self.n, create_using=nx.MultiGraph())
        for u in sorted(self.graph.nodes()):
            j = self.neigh_offset[u]
            while self.neigh_list[j] >= 0:
                v = self.neigh_list[j]
                if u < v:
                    self.graph.add_edge(u, v)
                j += 1
            
    def prepare_state(self):

        state = ffi.new("ising_state *", {
            "n": self.n,
            "field": self.field,
            "T": self.T,
            "seed": self.seed,
            "spins": ffi.cast("int8_t *", self.spins.ctypes.data),
            "neigh_list": ffi.cast("int32_t *", self.neigh_list.ctypes.data),
            "neigh_offset": ffi.cast("int32_t *", self.neigh_offset.ctypes.data),
            "degree": ffi.cast("int32_t *", self.degree.ctypes.data),
            "degree_sum": ffi.cast("int32_t *", self.degree_sum.ctypes.data),
            })
        return state

    def mc_sweep(self):

        assert self.neigh_list is not None

        state = self.prepare_state()
        r = cising.ising_mc_sweep(state)
        self.seed = state.seed
        return r

    def max_cluster(self, value=-1, edge_prob=1.0):

        assert self.neigh_list is not None

        state = self.prepare_state()
        r = cising.ising_max_cluster(state, value, edge_prob)
        self.seed = state.seed
        return r

    def mc_sweep_and_max_cluster(self, sweeps=10, value=-1, edge_prob=1.0):

        assert self.neigh_list is not None

        state = self.prepare_state()
        r = cising.ising_sweep_and_max_cluster(state, sweeps, value, edge_prob)
        self.seed = state.seed
        return r


def test():
    n = 1000
    spins = np.array([1] * n, dtype='int8')
    #g = nx.grid_graph(dim=[30,30], periodic=True)
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    print(g.size(), g.order())

    for i in range(1000):
        s = IsingState(spins, g, 1.5, 3.0)

    print(s.mc_sweep())
    print(s.max_cluster(-1, 1.0))

