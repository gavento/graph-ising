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

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;

uint64_t update_ns = 0;
uint64_t update_count = 0;
uint64_t cluster_ns = 0;
uint64_t cluster_count = 0;

#define START_TIMER() \
    struct timespec _timer_start; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timer_start);

#define STOP_TIMER(prefix, count) \
    struct timespec _timer_stop; \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timer_stop); \
    prefix##_count += count; \
    uint64_t _timer_secs = _timer_stop.tv_sec - _timer_start.tv_sec; \
    prefix##_ns += _timer_secs * 1000000000 + (_timer_stop.tv_nsec - _timer_start.tv_nsec);


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
 * Compute the Hamiltonian of the state.
 */
double ising_hamiltonian(ising_state *s, double F, double J)
{
    float H = 0.0;
    for (index_t v = 0; v < s->n; v++) {
        spin_t spin = s->spins[v];
        H -= spin * F;
        for (index_t i = 0; i < s->degree[v]; i++) {
            index_t u = s->neigh_list[s->neigh_offset[v] + i];
            if (u > v) {
                H -= J * spin * s->spins[u];
            }
        }
    }
    return H;
}


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

    double deltaE = -2.0 * s->spins[index] * (s->field + sum);
//    double deltaE = (1 + 2 * s->spins[index]) * (s->field - sum); <--- BUG: WHY ?!?
    if (deltaE > 0) {
        s->spins[index] = - s->spins[index];
        flipped = 1;
    } else {
        double p = get_rand_01(&s->seed);
        assert(p < 1.0);
        assert(p >= 0.0);
        if (p < exp(deltaE / s->T)) {
            s->spins[index] = - s->spins[index];
            flipped = 1;
        }
    }
    return flipped;
}


/*
 * Update a given number of spins choosing every one randomly, updating the state seed.
 * The spins to update are chosen randomly and independently (unlike in case of sweeps).
 * Returns the total number of flipped spins.
 */
index_t ising_mc_update_random(ising_state *s, index_t updates)
{
    START_TIMER();
    index_t flipped = 0;

    for (index_t i = 0; i < updates; i++) {
        index_t spin = get_rand(&s->seed) % s->n;
        flipped += ising_mc_update(s, spin);
    }

    STOP_TIMER(update, updates);
    return flipped;
}


/*
 * Update all the spins using a random permutation 'sweeps' times, updating the state seed.
 * Returns the number of flipped spins.
 * 
 * Note: When used directly does not update update_ns global stats.
 */
index_t ising_mc_sweep(ising_state *s, index_t sweeps)
{
    index_t flipped = 0;
    index_t *perm = alloca(sizeof(index_t[s->n]));

    for (int swi = 0; swi < sweeps; swi++) {
        get_rand_perm(s->n, perm, &s->seed);
        for (index_t i = 0; i < s->n; i++) {
            flipped += ising_mc_update(s, perm[i]);
        }
    }

    return flipped;
}


/*
 * Update a given number of spins using a part of a random permutation, updating the state seed.
 * When updates > N spins, first does several full sweeps and then updates the remaining
 * (updates % N) spins.
 * Returns the total number of flipped spins.
 */
index_t ising_mc_sweep_partial(ising_state *s, index_t updates)
{
    index_t flipped = 0;
    START_TIMER();

    if (updates >= s->n) {
        flipped += ising_mc_sweep(s, updates / s->n);
        updates = updates % s->n;
    }

    if (updates > 0) {
        index_t *perm = alloca(sizeof(index_t[s->n]));
        get_rand_perm(s->n, perm, &s->seed);
        for (index_t i = 0; i < updates; i++) {
            flipped += ising_mc_update(s, perm[i]);
        }
    }

    STOP_TIMER(update, updates);
    return flipped;
}


/*
 * Recursive helper to compute the statistics of the cluster containing given point using DFS.
 * The cluster starts at spin 'v' and expands to adjacent spins of the same value.
 * Every edge is present with probability edge_prob. Does NOT update the seed.
 *
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

        if ((edge_prob < 1.0) && (!get_rand_edge_presence(v, u, edge_prob, s->seed))) {
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
 * Only consider clusters of given value.
 *
 * Every edge is present with probability edge_prob.
 * Does NOT update the seed (consequent runs will give the same result).
 *
 * Computes all cluster statistics into 'stats' if not NULL.
 *
 * Warning: May overflow the stack during recursion.
 */
index_t ising_max_cluster_once(ising_state *s, spin_t value, double edge_prob, ising_cluster_stats *max_stats, uint8_t *out_mask)
{
    START_TIMER();
    index_t *visited = memset(alloca(sizeof(index_t[s->n])), 0, sizeof(index_t[s->n]));
    ising_cluster_stats cur_stats;
    if (max_stats)
        memset(max_stats, 0, sizeof(ising_cluster_stats));
    index_t max_size = 0, size, max_mark = 1;
    rand_t saved_seed = s->seed;

    for (index_t v = 0; v < s->n; v++) {

        if ((visited[v] == 0) && (s->spins[v] == value)) {

            index_t mark = v + 1;
            if (max_stats) {
                memset(&cur_stats, 0, sizeof(cur_stats));
                size = ising_max_cluster_visit(s, v, mark, visited, edge_prob, &cur_stats);
            } else {
                size = ising_max_cluster_visit(s, v, mark, visited, edge_prob, NULL);
            }
            if (size > max_size) {
                max_mark = mark;
                max_size = size;
                if (max_stats)
                    memcpy(max_stats, &cur_stats, sizeof(cur_stats));
            }
        }
    }

    if (max_stats) {
        assert(max_size == max_stats->v_in);
    }
    if (out_mask) {
        for (index_t v = 0; v < s->n; v++) {
            if ((visited[v] == max_mark) && (s->spins[v] == value)) {
                out_mask[v] = 1;
            } else {
                out_mask[v] = 0;
            }
        }
    }

    STOP_TIMER(cluster, s->n);
    assert(saved_seed == s->seed);
    return max_size;
}


/*
 * Measure the cluster stats 'measure' times, summing all the numbers.
 *
 * The seed is updated for all the sweeps, not for measurements (but they are performed with
 * different seeds).
 *
 * Returns max. cluster size (resp. their sum if measure > 1) after the sweeps.
 * When max_stats != NULL, store cluster maximums (resp. their sums) there.
 * 
 * If non-NULL, out_mask must be uint8_t[N] and will contain 1s on the largest cluster
 * from the last run.
 */
index_t ising_max_cluster_multi(ising_state *s, uint32_t measure, spin_t value,
                          double edge_prob, ising_cluster_stats *max_stats, uint8_t *out_mask)
{
    index_t sum = 0;
    ising_cluster_stats temp_max_stats;

    rand_t saved_seed = s->seed; // Save before measuring
    for (int i = 0; i < measure; i++) {

        if (max_stats) {
            memset(&temp_max_stats, 0, sizeof(ising_cluster_stats));
            sum += ising_max_cluster_once(s, value, edge_prob, &temp_max_stats, out_mask);
#define MSUM(attr) max_stats->attr += temp_max_stats.attr
            MSUM(v_in); MSUM(e_in); MSUM(e_border); MSUM(v_out_border); MSUM(v_in_border);
#undef MSUM
        } else {
            sum += ising_max_cluster_once(s, value, edge_prob, NULL, out_mask);
        }
        get_rand(&s->seed); // Advance 'temporary' rand seed (restored below)
    }
    s->seed = saved_seed; // Restore seed

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

    for (int i = 0; i < 100; i++) {
        index_t flips = ising_mc_sweep(&s, 1);
        index_t csize = ising_max_cluster_once(&s, -1, 1.0, &stats, NULL);
        assert(csize == stats.v_in);
        printf("Sweep: %d   flips: %d   stats: v_in=%d v_out_border=%d v_in_border=%d e_in=%d, e_border=%d\n",
               i, flips, stats.v_in, stats.v_out_border, stats.v_in_border, stats.e_in, stats.e_border);
    }


    return 0;
}
