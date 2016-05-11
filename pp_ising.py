import pp
import ising
import networkx as nx

def test():
    n = 10
    popsize = 1
    sweeps = 1
    iters = 1

    job_server = pp.Server(secret='jfhsa..gu=kawiue73du')

    print("Gen g ...")
    g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)

    print("Gen pop ...")
    pop = [ising.IsingState(g, seed=i, T=3.0, field=1.5) for i in range(popsize)]

    for i in range(iters):
        print("Run MC (#%d) ..." % i)
        jobs = [job_server.submit(state.mc_sweep_and_max_cluster, (sweeps, ), (), ()) for state in pop]
        for j in jobs:
            print(j())
    
test()
