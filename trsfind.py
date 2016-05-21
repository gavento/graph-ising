import concurrent.futures
import ising
import networkx as nx
import numpy as np
import time


def do_job(state, sweeps):
    return state.mc_sweep_and_max_cluster(sweeps)


def test2():
    n = 1000
    popsize = 200
    sweeps = 5
   # seed 1, iters = 1430
    iters = 1430

    print("Gen g ...")
    t0 = time.time()
    # g = nx.random_graphs.powerlaw_cluster_graph(n, 10, 0.1, 42)
    g=nx.read_gexf("test1000.gexf", node_type=int, relabel=False, version='1.1draft')

    print("Gen inistate ...")
    
    for iniseed in range(30) :

            inistate = ising.IsingState(graph=g, seed=iniseed, T=4.0, field=7.8) 
	    state = inistate.copy()
            
            print("Find transition state of inistate #",iniseed)

            for i in range(10000):
                m = do_job(state, sweeps)
                if state.max_cluster().v_in > 600 :
		# guess TS is 1*sweeps back
			print(state.max_cluster())
		    	iters = i -1 
                        print(iters)
			break

	    state= inistate.copy()

            print("Evolve again inistate #",iniseed)	   
            for i in range(iters):
                m = do_job(state, sweeps)

	    print(i)
            print(state.max_cluster())
            

            pop = [ state.copy() for i in range(popsize)]
           
	    
            print("Populate and MC")
            
            basin_a=0
            basin_b=0
            scount=1

            for state in pop:
                    state.seed=scount
                    scount+=1
                    m = do_job(state, 20)
		    print(state.max_cluster())
                    if state.max_cluster().v_in > n*0.9 :
                        basin_b+=1
                    elif state.max_cluster().v_in < n*0.1 :
                        basin_a+=1
            
            print("a b ",(basin_a,basin_b))             
                                     

    print("  Time: %f" % (time.time() - t0))






def test():

    # Test 0
    s = ising.IsingState(n=5, spins=[-1, -1, -1, 1, 1])
    s.set_edge_list([(0,1), (1,2), (2,3), (3,4), (4,0)])
    print(s.neigh_list)
    print(s.neigh_offset)
    print(s.degree)
    stat = s.max_cluster()
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
            m = np.mean([do_job(state, sweeps) for state in pop])

    # Multithreaded
    if True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(iters):
                jobs = [executor.submit(do_job, state, sweeps) for state in pop]
                m = np.mean([j.result().v_in for j in jobs])

    state = pop[0]
    print(state.mc_sweep_and_max_cluster(sweeps=0, measurements=1))
    print(state.mc_sweep_and_max_cluster(sweeps=0, measurements=1, edge_prob=0.8))
    print(state.mc_sweep_and_max_cluster(sweeps=0, measurements=100, edge_prob=0.8))
    print(state.mc_sweep_and_max_cluster(sweeps=100))
    print(state.max_cluster(edge_prob=0.8))


    print("  Time: %f" % (time.time() - t0))


test2()
