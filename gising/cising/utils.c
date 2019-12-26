#include "lib.h"

/*
 * Global counters, used for statistics.
 */
uint64_t update_ns = 0;
uint64_t update_count = 0;
uint64_t cluster_ns = 0;
uint64_t cluster_count = 0;


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
    for (index_t i = 0; i < n; i++)
    {
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
