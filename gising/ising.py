from cffi import FFI
import numpy as np
import networkx as nx
import json
import pickle


cising = None
ffi = None

def load_ffi():

    global cising, ffi

    ffi = FFI()
    ffi.cdef(
"""
typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;

typedef struct {
    index_t v_in;           // Vertices inside the cluster
    index_t v_out_border;   // Vertices adjacent to the cluster
    index_t v_in_border;    // Inner vertices adjacent to outside
    index_t e_in;           // Edges inside the cluster
    index_t e_border;       // Edges going out
} ising_cluster_stats;

typedef struct {
    index_t n;           // Number of spins (vertices)
    spin_t *spins;       // Values of spins
    double field;        // External dield
    double T;            // Temperature
    rand_t seed;         // Random seed. Modified with computation.
    index_t *neigh_list; // Neighbor lists for all vertices. Every list is -1 terminated.
    index_t *neigh_offset; // For every node starting offset in neigh_list
    index_t *degree;     // Node degrees (informational)
    index_t *degree_sum; // Sum of all degrees up to and incl. this node
			 // (for random edge generation)
} ising_state;

inline index_t ising_mc_update(ising_state *s, index_t index);
index_t ising_mc_update_random(ising_state *s, index_t updates);

index_t ising_mc_sweep(ising_state *s, index_t sweeps);
index_t ising_mc_sweep_partial(ising_state *s, index_t updates);

index_t ising_max_cluster_multi(ising_state *s, uint32_t measure, spin_t value,
                                double edge_prob, ising_cluster_stats *max_stats);

""")

    cising = ffi.dlopen('./_cising.so')

load_ffi()


class ClusterStats(object):

    def __init__(self, ising_cluster_stats, divide=1.0):
        self.v_in = ising_cluster_stats.v_in / divide
        self.v_in_border = ising_cluster_stats.v_in_border / divide
        self.v_out_border = ising_cluster_stats.v_out_border / divide
        self.e_in = ising_cluster_stats.e_in / divide
        self.e_border = ising_cluster_stats.e_border / divide


    def __repr__(self):

        return "<Stats: v_in=%s, v_in_border=%s, v_out_border=%s, e_in=%s, e_border=%s >" % (
                self.v_in, self.v_in_border, self.v_out_border, self.e_in, self.e_border)


    def __eq__(self, other):

        assert isinstance(other, ClusterStats)
        return (
                self.v_in == other.v_in and 
                self.v_in_border == other.v_in_border and 
                self.v_out_border == other.v_out_border and 
                self.e_in == other.e_in and 
                self.e_border == other.e_border
                )


class IsingState(object):

    def __init__(self, n=None, graph=None, spins=None, seed=42, field=0.0, T=1.0):

        if spins is None:
            if n != None:
                self.spins = np.ones([n], dtype='int8')
            elif graph != None:
                self.spins = np.ones([graph.order()], dtype='int8')
            else:
                raise ValueError('Provide n, graph or spins')
        else:
            self.spins = np.array(spins, dtype='int8')

        self.n = self.spins.shape[0]
        if n != None:
            assert n == self.n

        self.field = field
        self.T = T
        self.seed = seed

        assert self.spins.dtype.name == 'int8'
        assert len(self.spins.shape) == 1

        self.neigh_list = None
        self.neigh_offset = None
        self.degree = None
        self.degree_sum = None

        if graph:
            self.set_nx_graph(graph)


    def __repr__(self):
        return "<IsingState %d spins, seed %d>" % (self.n, self.seed)


    def __eq__(self, other):
        assert isinstance(other, IsingState)
        return (self.n == other.n and 
                self.seed == other.seed and
                self.T == other.T and
                self.field == other.field and
                all(self.spins == other.spins) and
                sorted(self.get_edge_list()) == sorted(other.get_edge_list()))


    def copy(self):

        IS = IsingState(spins=self.spins, seed=self.seed, field=self.field, T=self.T)
        IS.neigh_list = self.neigh_list.copy()
        IS.neigh_offset = self.neigh_offset.copy()
        IS.degree = self.degree.copy()
        IS.degree_sum = self.degree_sum.copy()
        return IS


    def save(self, fname):

        with open(fname, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, fname):

        with open(fname, 'rb') as f:
            return pickle.load(f)


    def get_edge_list(self):
        """
        Return a list of edges [(u,v), ...] based on the internal
        graph representation, every u-v edge present just once with u<v.
        """
        el = []
        for v in range(self.n):
            for i in range(self.degree[v]):
                u = self.neigh_list[self.neigh_offset[v] + i]
                if u < v:
                    el.append((u, v))
        return el


    def get_nx_graph(self):
        """
        Construct a NetworkX graph from the internal
        representation.
        """
        G = nx.empty_graph(self.n, create_using=nx.MultiGraph())
        G.add_edges_from(self.get_edge_list(), )
        return G


    def set_edge_list(self, edge_list):
        """
        Set the internal graph to a list of edges [(u,v), ...].
        Every u-v edge shoul appear only once (avoid [(u,v), (v,u), ...]).
        The graph is assumed to have the same number of vertices.
        The list may also be a set, the edges may be also sets.
        """

        self.degree = np.zeros([self.n], dtype='int32')

        edge_list = list(edge_list)
        for e in edge_list:
            u, v = tuple(e)
            self.degree[u] += 1
            self.degree[v] += 1
        
        self.degree_sum = np.cumsum(self.degree, out=self.degree_sum, dtype='int32')
        self.neigh_offset = np.array(np.concatenate([[0], self.degree_sum[:-1]]), dtype='int32')
        self.neigh_list = (-1) * np.ones([self.degree_sum[-1]], dtype='int32')

        offsets = self.neigh_offset.copy()
        for e in edge_list:
            u, v = tuple(e)
            self.neigh_list[offsets[u]] = v
            offsets[u] += 1;
            self.neigh_list[offsets[v]] = u
            offsets[v] += 1;

        assert all(offsets == self.degree_sum)
   

    def set_nx_graph(self, G):
        """
        Set the interbal list to a given graph.
        Needs to have integer vertices 0 .. N-1.
        """
        assert set(G.nodes()) == set(range(self.n))
        self.set_edge_list(G.edges())

            
    def _prepare_state(self):

        assert self.spins.dtype == 'int8'
        assert self.neigh_list.dtype == 'int32'
        assert self.neigh_offset.dtype == 'int32'
        assert self.degree.dtype == 'int32'
        assert self.degree_sum.dtype == 'int32'
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


    def mc_sweep(self, sweeps=1, updates=0):
        """
        Run `sweeps` full sweeps over the spins, each in random permutation order,
        and then `update` spin updates using a part of a permutation, updating the
        random seed every time.

        Returns the number of flips.
        """

        assert self.neigh_list is not None

        state = self._prepare_state()
        r = cising.ising_mc_sweep_partial(state, sweeps * self.n + updates)
        self.seed = state.seed

        return r


    def mc_random_update(self, updates=1):
        """
        Run `updates` of randomly chosen spins (independently, with replacement),
        updating the random seed every time.

        Returns the number of flips.
        """

        assert self.neigh_list is not None

        state = self._prepare_state()
        r = cising.ising_mc_update_random(state, updates)
        self.seed = state.seed

        return r



    def mc_max_cluster(self, measures=1, value=-1, edge_prob=1.0):
        """
        Measure maximal cluster statistics `measure` times (only considering spins with `value`)
        and return the averaged `ClusterStats`.

        This does not change the state seed, but if measures > 1,
        the measurements will be performed with different (temporary) seeds.
        """

        assert self.neigh_list is not None
        measures = int(measures)
        assert measures >= 1

        state = self._prepare_state()
        sum_max_stats = ffi.new("ising_cluster_stats *")
        r = cising.ising_max_cluster_multi(state, measures, value, edge_prob, sum_max_stats)
        self.seed = state.seed

        assert r == sum_max_stats.v_in
        return ClusterStats(sum_max_stats, divide=float(measures))


def test():

    n = 1000
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    print(g.size(), g.order())

    s = IsingState(graph=g, field=1.5, T=8.0)

    s1 = s.copy()
    s1.mc_random_update(10000)
    r1 = s1.mc_max_cluster(measures=3, edge_prob=0.9)

    s2 = s.copy()
    s2.mc_random_update(10000)
    r2 = s2.mc_max_cluster(measures=3, edge_prob=0.9)

    assert s1 == s2
    assert s1 != s
    assert r1 == r2

    s3 = s.copy()
    s3.mc_sweep(50)
    s3.mc_sweep(50)
    r3 = s3.mc_max_cluster(measures=1, edge_prob=1.0)
    
    s4 = s.copy()
    s4.mc_sweep(100)
    r4 = s4.mc_max_cluster(measures=1, edge_prob=1.0)

    assert s3 == s4
    assert s3 != s
    assert r3 == r4


