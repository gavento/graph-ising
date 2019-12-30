#include "lib.h"


/*
 * Compute the Hamiltonian of the state.
 */
double ising_hamiltonian(ising_state *s, double F, double J)
{
    float H = 0.0;
    for (index_t v = 0; v < s->g.n; v++)
    {
        spin_t spin = s->spins[v];
        assert(spin == -1 || spin == 1);
        H -= spin * F;
        for (index_t i = 0; i < s->g.degree[v]; i++)
        {
            index_t u = s->g.neigh_list[s->g.neigh_offset[v] + i];
            if (u > v)
            {
                H -= J * spin * s->spins[u];
            }
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
    for (index_t i = 0; i < s->g.degree[index]; i++)
    {
        index_t u = s->g.neigh_list[s->g.neigh_offset[index] + i];
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

/*
 * Update a given number of spins choosing every one randomly, updating the state seed.
 * The spins to update are chosen randomly and independently (unlike in case of sweeps).
 * Returns the total number of flipped spins.
 */
index_t ising_mc_update_random(ising_state *s, index_t updates)
{
    START_TIMER();
    index_t flipped = 0;

    for (index_t i = 0; i < updates; i++)
    {
        index_t spin = get_rand(&s->seed) % s->g.n;
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
    index_t *perm = alloca(sizeof(index_t[s->g.n]));

    for (int swi = 0; swi < sweeps; swi++)
    {
        get_rand_perm(s->g.n, perm, &s->seed);
        for (index_t i = 0; i < s->g.n; i++)
        {
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

    if (updates >= s->g.n)
    {
        flipped += ising_mc_sweep(s, updates / s->g.n);
        updates = updates % s->g.n;
    }

    if (updates > 0)
    {
        index_t *perm = alloca(sizeof(index_t[s->g.n]));
        get_rand_perm(s->g.n, perm, &s->seed);
        for (index_t i = 0; i < updates; i++)
        {
            flipped += ising_mc_update(s, perm[i]);
        }
    }

    STOP_TIMER(update, updates);
    return flipped;
}

/*
 * Updates a random spin until the up spins are <low, >=hi or timeout updates have been attempted.
 * Returns the number of flipped spins.
 */
index_t update_until_spincount(ising_state *s, index_t low, index_t hi, uint64_t timeout)
{
    START_TIMER();
    index_t flipped = 0;
    uint64_t i;
    for (i = 0; i < timeout; i++)
    {
        if (s->spins_up < low || s->spins_up >= hi)
            break;

        index_t spin = get_rand(&s->seed) % s->g.n;
        flipped += ising_mc_update(s, spin);
    }
    STOP_TIMER(update, i);
    return flipped;
}
