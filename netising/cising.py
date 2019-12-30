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

//// Graph

typedef struct
{
    index_t n;              // Number of spins (vertices)
    index_t m;              // Number of edges 
    index_t **neigh_list;   // Neighbor lists for all vertices.
                            // Every list is degree[v] long and has capacity neigh_cap[i].
    index_t *neigh_cap;     // Capacity of the neighbor lists
    index_t *degree;        // Node degrees
} ising_graph;

//// Ising

typedef struct
{
    index_t n; // Number of spins (vertices)
    ising_graph *g;
    spin_t *spins;    // Values of spins (-1, 1)
    double field;     // External field
    double T;         // Temperature
    rand_t seed;      // Random seed. Modified with computation.
    index_t spins_up; // Number of +1 spins (can be absolute or relative, updated only +-1).
    index_t updates;  // Attempted spin updates
} ising_state;

double ising_hamiltonian(ising_state *s, double F, double J);
index_t ising_mc_update(ising_state *s, index_t index);
index_t ising_mc_update_random(ising_state *s, index_t updates);
index_t ising_mc_sweep(ising_state *s, index_t sweeps);
index_t ising_mc_sweep_partial(ising_state *s, index_t updates);
index_t ising_update_until_spincount(ising_state *s, index_t low, index_t hi, uint64_t max_steps);


//// Games

typedef double ising_game_def[2][2][2];

typedef struct
{
    index_t n;            // Number of spins (vertices)
    ising_graph *g;       // Player network
    ising_game_def *game; // The game def. matrix as [p0_act][p1_act][payoff_for_whom]
    spin_t *states;       // Player actions (0, 1)
    double field;         // External field (preference for 1-states)
    double T;             // Temperature
    rand_t seed;          // Random seed. Modified with computation.
    index_t states_1;     // Number of 1 spins (can be absolute or relative, updated only +-1).
    index_t updates;      // Total attempted state updates
} ising_game_state;

double ising_game_hamiltonian(const ising_game_state *s);
index_t ising_game_mc_update(ising_game_state *s, index_t index);
index_t ising_game_mc_update_random(ising_game_state *s, index_t updates);
index_t ising_game_update_until_1count(ising_game_state *s, index_t low, index_t hi, uint64_t max_steps);


//// Clustering

typedef struct {
    index_t v_in;           // Vertices inside the cluster
    index_t v_out_border;   // Vertices adjacent to the cluster
    index_t v_in_border;    // Inner vertices adjacent to outside
    index_t e_in;           // Edges inside the cluster
    index_t e_border;       // Edges going out
} ising_cluster_stats;

index_t ising_max_cluster_multi(ising_state *s, uint32_t measure, spin_t value,
                                double edge_prob, ising_cluster_stats *max_stats,
                                uint8_t *out_mask);


""")

    so_path = os.path.join(os.path.dirname(__file__), '_cising.so')
    cising = ffi.dlopen(so_path)


load_ffi()
