import concurrent.futures
import ising
import networkx as nx
import numpy as np
import time


def do_job(state, sweeps):
    return state.mc_sweep_and_max_cluster(sweeps)


def test():
    n = 10000
    popsize = 100
    sweeps = 10
    iters = 100

    print("Gen g ...")
    t0 = time.time()
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    print("  Time: %f" % (time.time() - t0))

    print("Gen pop ...")
    t0 = time.time()
    pop = [ising.IsingState(g, seed=i, T=3.0, field=1.5, defer_init=True) for i in range(popsize)]
    print("  Time: %f" % (time.time() - t0))

    print("Init pop ...") # defer_init=True and separate state.graph_to_internal() to measure time
    t0 = time.time()
    for state in pop:
        state.graph_to_internal()
    print("  Time: %f" % (time.time() - t0))


    print("Run MCs ...")
    t0 = time.time()

    # One thread
    if False:
        for i in range(iters):
            m = np.mean([do_job(state, sweeps) for state in pop])

    # Multithreaded
    if True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for i in range(iters):
                jobs = [executor.submit(do_job, state, sweeps) for state in pop]
                m = np.mean([j.result() for j in jobs])

    print("  Time: %f" % (time.time() - t0))


test()
