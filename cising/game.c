#include "lib.h"

// typedef struct
// {
//     index_t n;            // Number of states (vertices)
//     ising_graph *g;       // Player network
//     ising_game_def *game; // The game def. matrix as [p0_act][p1_act][payoff_for_whom]
//     spin_t *states;       // Player actions (0, 1)
//     double field;         // External field (preference for 1-states)
//     double T;             // Temperature
//     rand_t seed;          // Random seed. Modified with computation.
//     index_t states_1;     // Number of 1 states (can be absolute or relative, updated only +-1).
//     index_t updates;      // Total attempted state updates
// } ising_game_state;

/*
 * Compute the Hamiltonian of the game state.
 */
double ising_game_hamiltonian(const ising_game_state *s)
{
    float H = 0.0;
    for (index_t v = 0; v < s->n; v++)
    {
        spin_t st = s->states[v];
        assert(st == 0 || st == 1);
        H -= st * s->field;
        for (index_t i = 0; i < s->g->degree[v]; i++)
        {
            index_t u = s->g->neigh_list[v][i];
            H -= (*s->game)[st][s->states[u]][0];
        }
    }
    return H;
}

/*
 * Update a single spin with MC rule, updating the state seed.
 */
index_t ising_game_mc_update(ising_game_state *s, index_t index)
{
    index_t flipped = 0;

    double deltaE = -2.0 * s->states[index] * s->field;
    for (index_t i = 0; i < s->g->degree[index]; i++)
    {
        index_t u = s->g->neigh_list[index][i];
        deltaE -= 2.0 * (*s->game)[s->states[index]][s->states[u]][0];
    }
    if (deltaE > 0)
    {
        flipped = 1;
    }
    else
    {
        double p = get_rand_01(&s->seed);
        assert(p < 1.0);
        assert(p >= 0.0);
        if (p < exp(deltaE / s->T))
            flipped = 1;
    }
    // printf("Updating %d from %d: flip=%d, deltaE=%.2f\n", index, s->states[index], flipped, deltaE);

    if (flipped)
    {
        s->states[index] = 1 - s->states[index];
        s->states_1 += 2 * s->states[index] - 1;
    }
    s->updates += 1;

    return flipped;
}

/*
 * Update a given number of states choosing every one randomly, updating the state seed.
 * The spins to update are chosen randomly and independently.
 * Returns the total number of flipped spins.
 */
index_t ising_game_mc_update_random(ising_game_state *s, index_t updates)
{
    START_TIMER();
    index_t flipped = 0;

    for (index_t i = 0; i < updates; i++)
    {
        index_t spin = get_rand(&s->seed) % s->n;
        flipped += ising_game_mc_update(s, spin);
    }

    STOP_TIMER(update, updates);
    return flipped;
}

/*
 * Updates a random spin until the up spins are <low, >=hi or max_step updates have been attempted.
 * Returns the number of flipped spins.
 */
index_t ising_game_update_until_1count(ising_game_state *s, index_t low, index_t hi, uint64_t max_steps)
{
    START_TIMER();
    index_t flipped = 0;
    uint64_t i;
    for (i = 0; i < max_steps; i++)
    {
        if (s->states_1 < low || s->states_1 >= hi)
            break;

        index_t spin = get_rand(&s->seed) % s->n;
        flipped += ising_game_mc_update(s, spin);
    }
    STOP_TIMER(update, i);
    return flipped;
}
