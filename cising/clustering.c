#include "lib.h"

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
static index_t ising_max_cluster_visit(ising_state *s, index_t v, index_t mark,
    index_t *visited,
    double edge_prob, ising_cluster_stats *stats)
{
    spin_t value = s->spins[v];
    visited[v] = mark;
    spin_t external_neighbours = 0;
    index_t size = 1;

    if (stats)
        stats->v_in++;

    for (index_t i = 0; i < s->g->degree[v]; i++)
    {

        index_t u = s->g->neigh_list[s->g->neigh_offset[v] + i];

        if ((edge_prob < 1.0) && (!get_rand_edge_presence(v, u, edge_prob, s->seed)))
        {
            continue; // Edge considered non-existent (is that right?)
        }

        if (s->spins[u] == value)
        { // Internal vertex

            if (visited[u] == mark)
            { // Already-visited inside vertex

                if (stats && v < u)
                    stats->e_in++;
            }
            else
            { // Unvisited internal vertex

                if (stats && v < u)
                    stats->e_in++;
                size += ising_max_cluster_visit(s, u, mark, visited, edge_prob, stats);
            }
        }
        else
        { // External vertex

            external_neighbours++;

            if (stats)
                stats->e_border++;

            if (visited[u] == mark)
            { // Already-noticed external vertex
                // Nothing :)
            }
            else
            { // Unvisited external vertex

                if (stats)
                    stats->v_out_border++;
                visited[u] = mark;
            }
        }
    }

    if (external_neighbours > 0 && stats)
        stats->v_in_border++;

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
index_t ising_max_cluster_once(ising_state *s, spin_t value, double edge_prob,
                               ising_cluster_stats *max_stats, uint8_t *out_mask)
{
    START_TIMER();
    index_t *visited = memset(alloca(sizeof(index_t[s->n])), 0, sizeof(index_t[s->n]));
    ising_cluster_stats cur_stats;
    if (max_stats)
        memset(max_stats, 0, sizeof(ising_cluster_stats));
    index_t max_size = 0, size, max_mark = 1;
    rand_t saved_seed = s->seed;

    for (index_t v = 0; v < s->n; v++)
    {

        if ((visited[v] == 0) && (s->spins[v] == value))
        {

            index_t mark = v + 1;
            if (max_stats)
            {
                memset(&cur_stats, 0, sizeof(cur_stats));
                size = ising_max_cluster_visit(s, v, mark, visited, edge_prob, &cur_stats);
            }
            else
            {
                size = ising_max_cluster_visit(s, v, mark, visited, edge_prob, NULL);
            }
            if (size > max_size)
            {
                max_mark = mark;
                max_size = size;
                if (max_stats)
                    memcpy(max_stats, &cur_stats, sizeof(cur_stats));
            }
        }
    }

    if (max_stats)
    {
        assert(max_size == max_stats->v_in);
    }
    if (out_mask)
    {
        for (index_t v = 0; v < s->n; v++)
        {
            if ((visited[v] == max_mark) && (s->spins[v] == value))
            {
                out_mask[v] = 1;
            }
            else
            {
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
    for (int i = 0; i < measure; i++)
    {

        if (max_stats)
        {
            memset(&temp_max_stats, 0, sizeof(ising_cluster_stats));
            sum += ising_max_cluster_once(s, value, edge_prob, &temp_max_stats, out_mask);
#define MSUM(attr) max_stats->attr += temp_max_stats.attr
            MSUM(v_in);
            MSUM(e_in);
            MSUM(e_border);
            MSUM(v_out_border);
            MSUM(v_in_border);
#undef MSUM
        }
        else
        {
            sum += ising_max_cluster_once(s, value, edge_prob, NULL, out_mask);
        }
        get_rand(&s->seed); // Advance 'temporary' rand seed (restored below)
    }
    s->seed = saved_seed; // Restore seed

    return sum;
}
