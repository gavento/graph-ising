#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <alloca.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

typedef uint64_t rand_t;
typedef int32_t index_t;
typedef int8_t spin_t;


uint32_t get_rand(rand_t *seed)
{
    *seed = (6364136223846793005ULL * (*seed)) + 1;
    return (*seed) >> 32;
}


double get_rand_01(rand_t *seed)
{
    return (double)(get_rand(seed) % 10000000) / 10000000.0;
}


void rand_perm(size_t n, index_t *result, rand_t *seed)
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


typedef struct {
    index_t n;
    spin_t *spins;
    double field;
    double T;
    index_t **neighbours; // -1 terminated neighbor list  
} ising_state;


inline index_t ising_mc_update(ising_state *s, index_t index, rand_t *seed)
{
    index_t flipped = 0;

    index_t sum = 0;
    for (index_t *p_neigh = s->neighbours[index]; *p_neigh >= 0; p_neigh++)
        sum += s->spins[*p_neigh];

    double deltaE = (1 + 2 * s->spins[index]) * (s->field - sum);
    if (deltaE > 0) {
        s->spins[index] = -s->spins[index];
        flipped = 1;
    } else {
        double p = get_rand_01(seed);
        if (p < exp(deltaE / s->T)) {
            s->spins[index] = -s->spins[index];
            flipped = 1;
        }
    }
    return flipped;
}


index_t ising_mc_sweep(ising_state *s, rand_t *seed)
{
    index_t flipped = 0;

    index_t *perm = alloca(sizeof(index_t[s->n]));
    rand_perm(s->n, perm, seed);
    for (index_t i = 0; i < s->n; i++) {
        flipped += ising_mc_update(s, perm[i], seed);
    }

    return flipped;
}

index_t ising_max_cluster_visit(ising_state *s, index_t index, uint8_t *visited, spin_t value, double edge_prob, rand_t *seed)
{
    index_t size = 1;
    visited[index] = 1;
    for (index_t *p_neigh = s->neighbours[index]; *p_neigh >= 0; p_neigh++) {
        index_t j = *p_neigh;
        if ((s->spins[j] == value) && (!visited[j]) && (get_rand_01(seed) < edge_prob))
            size += ising_max_cluster_visit(s, j, visited, value, edge_prob, seed);
    }
    return size;
}

index_t ising_max_cluster(ising_state *s, spin_t value, double edge_prob, rand_t *seed)
{
    uint8_t *visited = memset(alloca(sizeof(int[s->n])), 0, sizeof(int[s->n]));
    index_t max_size = 0;

    for (index_t i = 0; i < s->n; i++) {
        if ((!visited[i]) && (s->spins[i] == value)) {
            index_t size = ising_max_cluster_visit(s, i, visited, value, edge_prob, seed);
            max_size = MAX(size, max_size);
        }
    }
    return max_size;
}

int main_test()
{
    const int n = 30000;
    spin_t spins[n];
    index_t *neighbours[n];
    index_t ndata[30 * n], *nn = ndata;
    rand_t mc_seed = 42;
    rand_t cluster_seed = 43;

    for(index_t u = 0; u < n; u++) {
        spins[u] = 1;
        neighbours[u] = nn;
        for (int d = -10; d <= 10; d++) {
            if (d != 0) {
                *nn = (u + d + n) % n;
                nn++;
            }
        }
        *nn = -1;
        nn++;
    }

    ising_state s = {
        .n = n,
        .spins = spins,
        .field = 3,
        .T = 15.,
        .neighbours = neighbours,
    };

    for (int i = 0; i < 2000; i++) {
        index_t flips = ising_mc_sweep(&s, &mc_seed);
        index_t csize = ising_max_cluster(&s, -1, 1, &cluster_seed);
        //printf("Sweep: %d   flips: %d   max size: %d\n", i, flips, csize);
    }
    return 0;
}
