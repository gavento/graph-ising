#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <alloca.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;


/*
 * Get a random cca 30-bit number while updating the seed.
 */
uint32_t get_rand(rand_t *seed)
{
    *seed = (6364136223846793005ULL * (*seed)) + 1;
    return (*seed) >> 32;
}


/*
 * Get a random 0-1 real number (7 varying digits) while updating the seed.
 */
double get_rand_01(rand_t *seed)
{
    return (double)(get_rand(seed) % 10000000) / 10000000.0;
}


/*
 * Generate random permutation of {0 .. n-1} while updating the seed.
 */
void get_rand_perm(size_t n, index_t *result, rand_t *seed)
{
    // Available indices
    index_t *avail = alloca(sizeof(index_t[n]));
    for (index_t i = 0; i < n; i++)
        avail[i] = i;
    // Select available indices
    for (index_t i = 0; i < n; i++) {
        index_t x = get_rand(seed) % (n - i);
        result[i] = avail[x];
        avail[x] = avail[n - i - 1];
    }
}


/*
 * Decide if edge u-v should be considered in clustering with prob. edge_prob.
 * Gives the same results with the same seed (which is not modified), so
 * edges may be considered multiple times. Gives the same results for u-v and v-u.
 */
int get_rand_edge_presence(index_t u, index_t v, double edge_prob, rand_t seed)
{
    if (u < v)
        return get_rand_edge_presence(v, u, edge_prob, seed);

    rand_t temp_seed = seed + (((rand_t)u) << 32) + v;
    get_rand(&temp_seed);
    return (get_rand_01(&temp_seed) < edge_prob);
}


/*
 * Structure describing one ising model state, including the graph.
 * The graph is encoded in neigh_list as:
 * [neighbors of vertex 0 ..., -1, neighbors of vertex 1 ..., -1, ...]
 * The array neigh_offset[v] is the start index of v's neighbourhood.
 */
typedef struct {
    index_t n;           // Number of spins (vertices)
    spin_t *spins;       // Values of spins
    double field;        // External dield
    double T;            // Temperature
    rand_t seed;         // Random seed. Modified with computation.
    index_t *neigh_list; // Neighbor lists for all vertices. Every list is degree[v] long.
    index_t *neigh_offset; // For every node starting offset in neigh_list
    index_t *degree;     // Node degrees (informational)
    index_t *degree_sum; // Sum of all degrees up to and incl. this node
                         // (for random edge generation)
} ising_state;


/*
 * Structure describing either statistics of one cluster, or maximum statistics
 * over all clusters (each maximum computed independently).
 */
typedef struct {
    index_t v_in;           // Vertices inside the cluster
    index_t v_out_border;   // Vertices adjacent to the cluster
    index_t v_in_border;    // Inner vertices adjacent to outside
    index_t e_in;           // Edges inside the cluster
    index_t e_border;       // Edges going out
} ising_cluster_stats;


/*
 * Update a single spin with MC rule, updating the state seed.
 */
inline index_t ising_mc_update(ising_state *s, index_t index)
{
    index_t flipped = 0;

    index_t sum = 0;
    for (index_t i = 0; i < s->degree[index]; i++) {
        index_t u = s->neigh_list[s->neigh_offset[index] + i];
        sum += s->spins[u];
    }

    double deltaE = (1 + 2 * s->spins[index]) * (s->field - sum);
    if (deltaE > 0) {
        s->spins[index] = - s->spins[index];
        flipped = 1;
    } else {
        double p = get_rand_01(&s->seed);
        if (p < exp(deltaE / s->T)) {
            s->spins[index] = - s->spins[index];
            flipped = 1;
        }
    }
    return flipped;
}


/*
 * Update all the spins using a random permutation, updating the state seed.
 */
index_t ising_mc_sweep(ising_state *s)
{
    index_t flipped = 0;

    index_t *perm = alloca(sizeof(index_t[s->n]));
    get_rand_perm(s->n, perm, &s->seed);
    for (index_t i = 0; i < s->n; i++) {
        flipped += ising_mc_update(s, perm[i]);
    }

    return flipped;
}


/*
 * Recursive helper to compute the statistics of the cluster containing given point using DFS.
 * The cluster starts at spin 'v' and expands to adjacent spins of the same value.
 * Every edge is present with probability edge_prob.
 * The auxiliary array 'visited' is assumed to contain no values 'mark'
 * before each clustre examination start.
 *
 * Computes all cluster statistics into 'stats' if not NULL,
 * returns the size of the cluster.
 *
 * Warning: May overflow the stack during recursion.
 */
index_t ising_max_cluster_visit(ising_state *s, index_t v, index_t mark, index_t *visited,
                                double edge_prob, ising_cluster_stats *stats)
{
    spin_t value = s->spins[v];
    visited[v] = mark;
    spin_t external_neighbours = 0;
    index_t size = 1;

    if (stats)
        stats->v_in ++;

    for (index_t i = 0; i < s->degree[v]; i++) {

        index_t u = s->neigh_list[s->neigh_offset[v] + i];

        if (!get_rand_edge_presence(v, u, edge_prob, s->seed)) {
            continue; // Edge considered non-existent (is that right?)
        }

        if (s->spins[u] == value) { // Internal vertex

            if (visited[u] == mark) { // Already-visited inside vertex

                if (stats && v < u) 
                    stats->e_in ++;

            } else { // Unvisited internal vertex

                if (stats && v < u)
                    stats->e_in ++;
                size += ising_max_cluster_visit(s, u, mark, visited, edge_prob, stats);

            }

        } else { // External vertex

            external_neighbours ++;

            if (stats)
                stats->e_border ++;

            if (visited[u] == mark) { // Already-noticed external vertex
                // Nothing :)
            } else { // Unvisited external vertex

                if (stats)
                    stats->v_out_border ++;
                visited[u] = mark;
            }

        }

    }

    if (external_neighbours > 0 && stats)
        stats->v_in_border ++;

    return size;
}


/*
 * Compute the statistics of all the clusters using ising_max_cluster_visit.
 * Only consider clusters of given value. Every edge is present with probability edge_prob.
 *
 * Computes all cluster statistics into 'stats' if not NULL.
 *
 * Warning: May overflow the stack during recursion.
 */
index_t ising_max_cluster(ising_state *s, spin_t value, double edge_prob, ising_cluster_stats *max_stats)
{
    index_t *visited = memset(alloca(sizeof(index_t[s->n])), 0, sizeof(index_t[s->n]));
    ising_cluster_stats cur_stats;
    if (max_stats)
        memset(max_stats, 0, sizeof(ising_cluster_stats));
    index_t max_size = 0, size;

    for (index_t v = 0; v < s->n; v++) {

        if ((visited[v] == 0) && (s->spins[v] == value)) {

            if (max_stats) {
                memset(&cur_stats, 0, sizeof(cur_stats));
                size = ising_max_cluster_visit(s, v, v + 1, visited, edge_prob, &cur_stats);
#define MSTAT(attr) max_stats->attr = MAX(max_stats->attr, cur_stats.attr)
                MSTAT(v_in); MSTAT(e_in); MSTAT(e_border); MSTAT(v_out_border); MSTAT(v_in_border);
#undef MSTAT
            } else {
                size = ising_max_cluster_visit(s, v, v + 1, visited, edge_prob, NULL);
            }
            max_size = MAX(size, max_size);

        }
    }

    if (max_stats) {
        assert(max_size == max_stats->v_in);
    }

    // Advance the seed (in case of recomputations)
    get_rand(&s->seed);
    return max_size;
}


/*
 * Perform sweep 'sweeps' times and measure the cluster stats 'measurements' times,
 * summing all the numbers. 
 * Returns max. cluster size (resp. their sum if measurements > 1) after the sweeps.
 * When max_stats != NULL, store cluster maximums (resp. their sums) there.
 */
index_t ising_sweep_and_max_cluster(ising_state *s, uint32_t sweeps, uint32_t measurements, spin_t value,
                                    double edge_prob, ising_cluster_stats *max_stats)
{

    for (int i = 0; i < sweeps; i++)
        ising_mc_sweep(s);

    index_t sum = 0;
    ising_cluster_stats temp_max_stats;

    for (int i = 0; i < measurements; i++) {

        if (max_stats) {
            memset(&temp_max_stats, 0, sizeof(ising_cluster_stats));
            sum += ising_max_cluster(s, value, edge_prob, &temp_max_stats);
#define MSUM(attr) max_stats->attr += temp_max_stats.attr
            MSUM(v_in); MSUM(e_in); MSUM(e_border); MSUM(v_out_border); MSUM(v_in_border);
#undef MSUM
        } else {
            sum += ising_max_cluster(s, value, edge_prob, NULL);
        }
    }

    return sum;
}

int main_test()
{
    const int n = 10000;

    ising_state s = {
        .n = n,
        .spins = malloc(sizeof(spin_t[n])),
        .field = 3,
        .T = 15.,
        .seed = 42,
        .neigh_list = malloc(sizeof(index_t[30 * n])),
        .neigh_offset = malloc(sizeof(index_t[n])),
        .degree = malloc(sizeof(index_t[n])),
        .degree_sum = malloc(sizeof(index_t[n])),
    };

    uint32_t nlpos = 0;
    for(index_t i = 0; i < n; i++) {
        s.spins[i] = 1;
        s.neigh_offset[i] = nlpos;
        for (int d = -10; d <= 10; d++) {
            if (d != 0) {
                s.neigh_list[nlpos++] = (i + d + n) % n;
            }
        }
        s.degree[i] = 20;
        s.degree_sum[i] = 20 * i;
    }

    ising_cluster_stats stats;

    for (int i = 0; i < 1000; i++) {
        index_t flips = ising_mc_sweep(&s);
        index_t csize = ising_max_cluster(&s, -1, 1.0, &stats);
        printf("Sweep: %d   flips: %d   stats: v_in=%d v_out_border=%d v_in_border=%d e_in=%d, e_border=%d\n",
               i, flips, stats.v_in, stats.v_out_border, stats.v_in_border, stats.e_in, stats.e_border);
    }


    return 0;
}
