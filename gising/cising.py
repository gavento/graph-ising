import os

from cffi import FFI

cising = None
ffi = None


def load_ffi():

    global cising, ffi

    ffi = FFI()
    ffi.cdef("""
typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;

uint64_t update_ns;
uint64_t update_count;
uint64_t cluster_ns;
uint64_t cluster_count;

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
    index_t spins_up;    // Number of +1 spins (can be absolute or relative, updated only +-1).
    index_t updates;     // Attempted spin updates
    index_t *neigh_list; // Neighbor lists for all vertices. Every list is degree[v] long.
    index_t *neigh_offset; // For every node starting offset in neigh_list
    index_t *degree;     // Node degrees (informational)
    index_t *degree_sum; // Sum of all degrees up to and incl. this node
                         // (for random edge generation)
} ising_state;

index_t ising_mc_update_random(ising_state *s, index_t updates);

double ising_hamiltonian(ising_state *s, double F, double J);

index_t ising_max_cluster_multi(ising_state *s, uint32_t measure, spin_t value,
                                double edge_prob, ising_cluster_stats *max_stats,
                                uint8_t *out_mask);

index_t update_until_spincount(ising_state *s, index_t low, index_t hi, uint64_t timeout);

""")

    so_path = os.path.join(os.path.dirname(__file__), '_cising.so')
    cising = ffi.dlopen(so_path)

load_ffi()