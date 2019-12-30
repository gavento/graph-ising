#include "lib.h"

int main()
{
    const int n = 10000;

    ising_graph g = {
        .n = n,
        .neigh_list = malloc(sizeof(index_t * [n])),
        .neigh_cap = malloc(sizeof(index_t[n])),
        .degree = malloc(sizeof(index_t[n])),
    };
    ising_state s = {
        .n = n,
        .g = &g,
        .spins = malloc(sizeof(spin_t[n])),
        .field = 3,
        .T = 15.,
        .seed = 42,
    };

    for (index_t i = 0; i < n; i++)
    {
        s.spins[i] = 1;
        s.g->neigh_list[i] = malloc(sizeof(index_t[n]));
        s.g->neigh_cap[i] = n;
        index_t *p = &s.g->neigh_list[i][0];
        for (int d = -10; d <= 10; d++)
        {
            if (d != 0)
            {
                *p = (i + d + n) % n;
                p++;
            }
        }
        s.g->degree[i] = 20;
    }

    ising_cluster_stats stats;

    for (int i = 0; i < 20; i++)
    {
        index_t flips = ising_mc_sweep(&s, 1);
        index_t csize = ising_max_cluster_once(&s, -1, 1.0, &stats, NULL);
        assert(csize == stats.v_in);
        printf("Sweep: %d   flips: %d   stats: v_in=%d v_out_border=%d v_in_border=%d e_in=%d, e_border=%d\n",
               i, flips, stats.v_in, stats.v_out_border, stats.v_in_border, stats.e_in, stats.e_border);
    }

    return 0;
}
