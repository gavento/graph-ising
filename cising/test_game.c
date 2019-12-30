#include "lib.h"

int main()
{
    const int n = 4;

    index_t ns[] = {1, 2, 3, 0, 2, 0, 1, 0};
    index_t ds[] = {3, 2, 2, 1};
    index_t *nsp[] = {ns + 0, ns + 3, ns + 5, ns + 7};
    ising_graph g = {
        .n = n,
        .neigh_list = nsp,
        .neigh_cap = ds,
        .degree = ds,
    };

    double game[2][2][2] = {{{0, 0}, {0, 0}},{{0, 0}, {0, 0}}};
    spin_t sps[] = {0, 0, 1, 1};
    ising_game_state s = {
        .n = n,
        .game = &game,
        .g = &g,
        .states = sps,
        .field = 3.,
        .T = 15.,
        .seed = 42,
        .states_1 = 2,
    };

    for (int i = 0; i < 20; i++)
    {
        index_t flips = ising_game_mc_update_random(&s, n);
        printf("Sweep: %d   flips=%d   count_1=%d   H=%.2f\n",
               i, flips, s.states_1, ising_game_hamiltonian(&s));
    }

    return 0;
}
