#!/usr/bin/env python

import bz2
import itertools
import json
import os
import pickle
import re
import signal

import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
import scipy as sp
import scipy.stats

from gising import utils
from gising.forward_flux import FFSampler
from gising.ising_state import (ClusterOrderIsingState, SpinCountIsingState, report_runtime_stats)


def main():
    parser = utils.default_parser()
    parser.add_argument("graph", metavar="GRAPHML", type=str, help="Use given GraphML file.")
    parser.add_argument("--samples",
                        "-s",
                        default=10,
                        type=int,
                        help="Samples required at each interface.")
    parser.add_argument("--Is",
                        "-I",
                        default=None,
                        type=int,
                        help="Interfaces to use (default: dynamic).")
    parser.add_argument("--Imin",
                        default=None,
                        type=int,
                        help="Interface A order param (default: from mesostable).")
    parser.add_argument("--Imax",
                        default=None,
                        type=int,
                        help="Interface B order param (default: from mesostable).")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
    parser.add_argument("--meso_time",
                        "-m",
                        default=1000.0,
                        type=float,
                        help="Time spent for mesostable state sampling.")
    parser.add_argument("--stop_rate",
                        default=None,
                        type=float,
                        help="If given, stop if rate goes below [given as 10-exponent].")
    parser.add_argument("--timeout",
                        default=10000.0,
                        type=float,
                        help="One simulation run timeout (in MC sweeps).")
    parser.add_argument("--cluster_samples",
                        "-C",
                        default=None,
                        type=int,
                        help="Cluster size samples to run.")
    parser.add_argument('--count_spins',
                        '-S',
                        action='store_true',
                        default=False,
                        help="Use spin count as the order param (default: largest cluster size)")
    parser.add_argument(
        "--fix_cluster_seed",
        default=42,
        type=int,
        help="Fix clustering seed (requires high -C for good results), 0 for random.")

    args = utils.init_experiment(parser)

    tee = utils.Tee(args.logfile)
    tee.start()

    with utils.timed(f"reading graph '{args.graph}'"):
        g = nx.read_graphml(args.graph)

    with utils.timed("create state"):
        if args.count_spins:
            state0 = SpinCountIsingState(g, T=args.T, F=args.F)
        else:
            raise NotImplementedError()
            # TODO: Special clustering options will go here
            state0 = ClusterOrderIsingState(g, T=args.T, F=args.F)

    chaotic = False

    if args.Imin is None:
        with utils.timed("sample mesostable for iface A"):
            spl = state0.sample_mesostable(progress=args.progress and tee.stderr,
                                           time=args.meso_time / 5,
                                           samples=min(int(args.meso_time), 10),
                                           trials=5)
            spl = spl[:, (spl.shape[1] // 2):]
            args.Imin = int(sp.stats.norm.ppf(1 - 1e-4, loc=np.mean(spl), scale=np.std(spl)))
            print(f"Selected LambdaA={args.Imin} based on {utils.stat_str(spl, True, prec=5)}")
            if args.Imax is not None and args.Imin >= args.Imax:
                chaotic = True
                print(
                    "Mesostable(-1) upper-limit above target order - divergent process or above crit. temp.?"
                )

    if args.Imax is None:
        with utils.timed("sample mesostable for iface B"):
            state1 = state0.copy()
            state1.set_spins(np.ones(state1.n))
            spl = state1.sample_mesostable(progress=args.progress and tee.stderr,
                                           time=args.meso_time / 5,
                                           samples=min(int(args.meso_time), 10),
                                           trials=5)
            spl = spl[:, (spl.shape[1] // 2):]
            args.Imax = int(sp.stats.norm.ppf(1e-4, loc=np.mean(spl), scale=np.std(spl)))
            print(f"Selected LambdaB={args.Imax} based on {utils.stat_str(spl, True, prec=5)}")
            if args.Imin >= args.Imax:
                chaotic = True
                print("Mesostable(+1) lower-limit below LambdaA - above crit. temp.?")

    if args.Is is not None:
        ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
        print(f"Interfaces: {np.array(ifs)[:20]} ...")
    else:
        ifs = [args.Imin, args.Imax]
        print(f"Interfaces: dynamic {ifs[0]} .. {ifs[1]}")

    ff = FFSampler([state0], ifs, iface_samples=args.samples)

    try:
        if not chaotic:
            with utils.timed(f"FF compute"):
                ff.compute(progress=args.progress and tee.stderr,
                           timeout=args.timeout,
                           dynamic_ifaces=args.Is is None,
                           stop_rate=args.stop_rate)
        else:
            print("... skiping ff.compute(), system chaotic")

        with utils.timed(f"write '{args.fbase + '.json.bz2'}'"):
            d = dict(
                Name=args.full_name,
                Comment=args.comment,
                LambdaA=int(ff.interfaces[0].order),
                Log10RateA=ff.interfaces[0].log10_rate,
                LambdaB=int(ff.interfaces[-1].order),
                Log10RateB=ff.interfaces[-1].log10_rate,
                CSize=ff.critical_order_param(),
                T=args.T,
                F=args.F,
                Graph=args.graph,
                Chaotic=chaotic,
                Param='UpSpins' if args.count_spins else 'UpCluster',
                Samples=args.samples,
                N=g.order(),
                M=g.size(),
                Orders=[int(iface.order) for iface in ff.interfaces],
                Clusters=[[int(x)
                           for x in iface.states[0].get_stats().mask]
                          for iface in ff.interfaces
                          if iface.states],
                Hamiltonians=[[s.get_hamiltonian() for s in iface.states] for iface in ff.interfaces
                             ],
                UpFlows=[
                    iface.up_flow()**(1 / (ff.interfaces[ino + 1].order - iface.order))
                    for ino, iface in enumerate(ff.interfaces[:-1])
                ],
                Log10Rates=[iface.log10_rate for iface in ff.interfaces],
            )
            with bz2.BZ2File(args.fbase + '.json.bz2', 'w') as jf:
                jf.write(json.dumps(d).encode('utf-8'))

        with utils.timed(f"write '{args.fbase + '.ffs.pickle.bz2'}'"):
            with bz2.BZ2File(args.fbase + '.ffs.pickle.bz2', 'w') as f:
                pickle.dump(ff, f)

    except KeyboardInterrupt:
        print("\nInterrupted, trying to still report anything already computed ...")

    ### Reportss in iface.states

    print(f"FF cising stats:\n{report_runtime_stats()}")

    print()
    print(
        f"Interface A at {ff.ifaceA.order}, rate 10^{ff.ifaceA.log10_rate:.3f}={10**ff.ifaceA.log10_rate:.5g}"
    )
    print(
        f"Interface B at {ff.ifaceB.order}, rate 10^{ff.ifaceB.log10_rate:.3f}={10**ff.ifaceB.log10_rate:.5g}"
    )
    nucleus = ff.critical_order_param()
    if nucleus is not None:
        print(f"Critical nucleus size at {nucleus:.5g}")
        with utils.timed(f'writing critical core sample to {args.fbase + "-core.graphml.bz2"}'):
            for iface in reversed(ff.interfaces):
                if iface.order < nucleus:
                    break
            print(f"  Writing samples from iface at {iface.order}")
            g2 = g.copy()
            state = iface.states[0]
            mask = state.get_stats().mask
            for vi, v in enumerate(state.nodes):
                if mask[vi]:
                    g2.nodes[v]['core'] = True
                else:
                    g2.nodes[v]['core'] = False
            nx.write_graphml(g2, args.fbase + "-core.graphml.bz2")

    print()
    print(f"Log in {args.fbase + '.log'}")
    print(f"Data in {args.fbase + '.json.bz2'} (use plot_ffs.py to plot)")


def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, handle_pdb)
    print(os.getpid())
    main()
