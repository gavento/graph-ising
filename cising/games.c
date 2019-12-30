#include "lib.h"

/*
 * Compute the Hamiltonian of the game state.
 */
double ising_game_hamiltonian(const ising_state *s, const ising_game_def gd)
{
    float H = 0.0;
    for (index_t v = 0; v < s->n; v++)
    {
        spin_t spin = s->spins[v];
        assert(spin == 0 || spin == 1);
        H -= spin * s->field;
        for (index_t i = 0; i < s->degree[v]; i++)
        {
            index_t u = s->neigh_list[s->neigh_offset[v] + i];
            H -= gd[spin][s->spins[u]][0];
        }
    }
    return H;
}

/*
 * Update a single spin with MC rule, updating the state seed.
 */
index_t ising_mc_update(ising_state *s, index_t index)
{
    index_t flipped = 0;

    index_t sum = 0;
    for (index_t i = 0; i < s->degree[index]; i++)
    {
        index_t u = s->neigh_list[s->neigh_offset[index] + i];
        sum += s->spins[u];
    }

    double deltaE = -2.0 * s->spins[index] * (s->field + sum);
    //    double deltaE = (1 + 2 * s->spins[index]) * (s->field - sum); <--- BUG: WHY ?!?
    if (deltaE > 0)
    {
        s->spins[index] = -s->spins[index];
        flipped = 1;
    }
    else
    {
        double p = get_rand_01(&s->seed);
        assert(p < 1.0);
        assert(p >= 0.0);
        if (p < exp(deltaE / s->T))
        {
            s->spins[index] = -s->spins[index];
            flipped = 1;
        }
    }

    if (flipped)
        s->spins_up += s->spins[index];
    s->updates += 1;

    return flipped;
}
