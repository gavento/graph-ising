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
    rand_t seed;
    index_t *neigh_list; // -1 terminated neighbor list
    index_t *neigh_offset; // for every node starting offset in neigh_list
    index_t *degree; // node degrees
    index_t *degree_sum; // sum of all degrees up to and incl. this node
} ising_state;




inline index_t ising_mc_update(ising_state *s, index_t index)
{
    index_t flipped = 0;

    index_t sum = 0;
    for (index_t *p_neigh = s->neigh_list + s->neigh_offset[index]; *p_neigh >= 0; p_neigh++)
        sum += s->spins[*p_neigh];

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


index_t ising_mc_sweep(ising_state *s)
{
    index_t flipped = 0;

    index_t *perm = alloca(sizeof(index_t[s->n]));
    rand_perm(s->n, perm, &s->seed);
    for (index_t i = 0; i < s->n; i++) {
        flipped += ising_mc_update(s, perm[i]);
    }

    return flipped;
}

index_t ising_max_cluster_visit(ising_state *s, index_t index, uint8_t *visited, spin_t value, double edge_prob)
{
    index_t size = 1;
    visited[index] = 1;
    for (index_t *p_neigh = s->neigh_list + s->neigh_offset[index]; *p_neigh >= 0; p_neigh++) {
        index_t j = *p_neigh;
        if ((s->spins[j] == value) && (!visited[j]) && (get_rand_01(&s->seed) < edge_prob))
            size += ising_max_cluster_visit(s, j, visited, value, edge_prob);
    }
    return size;
}

index_t ising_max_cluster(ising_state *s, spin_t value, double edge_prob)
{
    uint8_t *visited = memset(alloca(sizeof(int[s->n])), 0, sizeof(int[s->n]));
    index_t max_size = 0;

    for (index_t i = 0; i < s->n; i++) {
        if ((!visited[i]) && (s->spins[i] == value)) {
            index_t size = ising_max_cluster_visit(s, i, visited, value, edge_prob);
            max_size = MAX(size, max_size);
        }
    }
    return max_size;
}

index_t ising_sweep_and_max_cluster(ising_state *s, uint32_t sweeps, spin_t value, double edge_prob)
{
    for (int i = 0; i < sweeps; i++)
        ising_mc_sweep(s);
    return ising_max_cluster(s, value, edge_prob);
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
        s.neigh_list[nlpos++] = -1;
    }

    for (int i = 0; i < 1000; i++) {
        index_t flips = ising_mc_sweep(&s);
//        index_t csize = ising_max_cluster(&s, -1, 1, &cluster_seed);
        //printf("Sweep: %d   flips: %d   max size: %d\n", i, flips, csize);
    }
    return 0;
}
