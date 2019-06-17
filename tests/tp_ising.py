import concurrent.futures
import ising
import networkx as nx
import numpy as np
import time


def do_job(state, sweeps):
    state.mc_sweep(sweeps)
    return state.mc_max_cluster()


def test():

    # Test 0
    s = ising.IsingState(n=5, spins=[-1, -1, -1, 1, 1])
    s.set_edge_list([(0,1), (1,2), (2,3), (3,4), (4,0)])
    print(s.neigh_list)
    print(s.neigh_offset)
    print(s.degree)
    stat = s.mc_max_cluster()
    assert stat.v_in == 3
    assert stat.v_in_border == 2
    assert stat.v_out_border == 2
    assert stat.e_in == 2
    assert stat.e_border == 2

    # Test 2
    n = 1000
    popsize = 1
    sweeps = 10
    iters = 200

    print("Gen g ...")
    t0 = time.time()
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    print("  Time: %f" % (time.time() - t0))

    print("Gen pop ...")
    t0 = time.time()
    pop = [ising.IsingState(graph=g, seed=i, T=6.0, field=1.5) for i in range(popsize)]
    print("  Time: %f" % (time.time() - t0))


    print("Run MCs ...")
    t0 = time.time()

    # One thread
    if False:
        for i in range(iters):
            m = np.mean([do_job(state, sweeps).v_in for state in pop])

    # Multithreaded
    if True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(iters):
                jobs = [executor.submit(do_job, state, sweeps) for state in pop]
                m = np.mean([j.result().v_in for j in jobs])

    state = pop[0]
    state.mc_sweep(sweeps=1000)
    print(state.mc_max_cluster(edge_prob=1.0))


    print("  Time: %f" % (time.time() - t0))


test()
