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
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)

    print("Gen pop ...")
    t0 = time.time()
    pop = [ising.IsingState(g, seed=i, T=3.0, field=1.5, defer_init=True) for i in range(popsize)]
    print("  Time: %f" % (time.time() - t0))

    print("Init pop ...")
    t0 = time.time()
    for state in pop:
        state.graph_to_internal()
    print("  Time: %f" % (time.time() - t0))


    print("Run MCs ...")
    t0 = time.time()

    if True:
        for i in range(iters):
            m = np.mean([do_job(state, sweeps) for state in pop])

    if False:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(iters):
#            print("Run MC (#%d) ..." % i)
                jobs = [executor.submit(do_job, state, sweeps) for state in pop]
                m = np.mean([j.result() for j in jobs])

    print("  Time: %f" % (time.time() - t0))


test()
