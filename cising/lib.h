#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <alloca.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;

/*
 * Structure describing a static network.
 * The graph is encoded in neigh_list as:
 * [neighbors of vertex 0 ..., -1 (one or many), neighbors of vertex 1 ..., -1 (one or many), ...]
 * The array neigh_offset[v] is the start index of v's neighbourhood.
 */

typedef struct
{
    index_t n;             // Number of spins (vertices)
    index_t m;             // Number of edges 
    index_t *neigh_list;   // Neighbor lists for all vertices. Every list is degree[v] long.
    index_t *neigh_offset; // For every node starting offset in neigh_list
    index_t *degree;       // Node degrees
} ising_graph;

typedef struct
{
    index_t n;                  // Number of spins (vertices)
    index_t m;                  // Number of edges 
    index_t *out_neigh_list;    // Out-neighbor lists for all vertices. Every list is out_degree[v] long.
    index_t *out_neigh_offset;  // For every node starting offset in out_neigh_list
    index_t *out_degree;        // Node out-degrees
    index_t *in_neigh_list;     // In-neighbor lists for all vertices. Every list is out_degree[v] long.
    index_t *in_neigh_offset;   // For every node starting offset in in_neigh_list
    index_t *in_degree;         // Node in-degrees
} ising_digraph;

/*
 * Structure describing one ising model state, including the graph.
 * The graph is encoded in neigh_list as:
 * [neighbors of vertex 0 ..., -1, neighbors of vertex 1 ..., -1, ...]
 * The array neigh_offset[v] is the start index of v's neighbourhood.
 */
typedef struct
{
    ising_graph *g;
    spin_t *spins;         // Values of spins (-1, 1) or game states (0, 1)
    double field;          // External field
    double T;              // Temperature
    rand_t seed;           // Random seed. Modified with computation.
    index_t spins_up;      // Number of +1 spins (can be absolute or relative, updated only +-1).
    index_t updates;       // Attempted spin updates
} ising_state;


/* 
 * cising.c *********************************************************
 */

double ising_hamiltonian(ising_state *s, double F, double J);
index_t ising_mc_update(ising_state *s, index_t index);
index_t ising_mc_update_random(ising_state *s, index_t updates);
index_t ising_mc_sweep(ising_state *s, index_t sweeps);
index_t ising_mc_sweep_partial(ising_state *s, index_t updates);
index_t update_until_spincount(ising_state *s, index_t low, index_t hi, uint64_t timeout);

/* 
 * games.c **********************************************************
 */

/*
 * indexed as [p0_action][p1_action][payoff_for_whom]
 */
typedef double ising_game_def[2][2][2];

/*
 * clustering.c *****************************************************
 */

/*
 * Structure describing either statistics of one cluster, or maximum statistics
 * over all clusters (each maximum computed independently).
 */
typedef struct
{
    index_t v_in;         // Vertices inside the cluster
    index_t v_out_border; // Vertices adjacent to the cluster
    index_t v_in_border;  // Inner vertices adjacent to outside
    index_t e_in;         // Edges inside the cluster
    index_t e_border;     // Edges going out
} ising_cluster_stats;

index_t ising_max_cluster_once(ising_state *s, spin_t value, double edge_prob,
                               ising_cluster_stats *max_stats, uint8_t *out_mask);
index_t ising_max_cluster_multi(ising_state *s, uint32_t measure, spin_t value,
                                double edge_prob, ising_cluster_stats *max_stats, uint8_t *out_mask);

/*
 * utils.c **********************************************************
 */

uint32_t get_rand(rand_t *seed);
double get_rand_01(rand_t *seed);
void get_rand_perm(size_t n, index_t *result, rand_t *seed);
int get_rand_edge_presence(index_t u, index_t v, double edge_prob, rand_t seed);

/*
 * Global counters, used for statistics.
 */
extern uint64_t update_ns;
extern uint64_t update_count;
extern uint64_t cluster_ns;
extern uint64_t cluster_count;

#define START_TIMER()             \
    struct timespec _timer_start; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timer_start);

#define STOP_TIMER(prefix, count)                                    \
    struct timespec _timer_stop;                                     \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timer_stop);           \
    prefix##_count += count;                                         \
    uint64_t _timer_secs = _timer_stop.tv_sec - _timer_start.tv_sec; \
    prefix##_ns += _timer_secs * 1000000000 + (_timer_stop.tv_nsec - _timer_start.tv_nsec);
