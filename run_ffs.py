#!/usr/bin/env python

import bz2
import itertools
import json
import pickle

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
                        type=float,
                        help="Interface A order param (default: from mesostable).")
    parser.add_argument("--Imax",
                        default=None,
                        type=float,
                        help="Interface B order param (default: from mesostable).")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
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

    if args.Imin is None:
        with utils.timed("sample mesostable for iface A"):
            spl = state0.sample_mesostable(progress=args.progress and tee.stderr)
            spl = spl[:, (spl.shape[1] * 2 // 3):]
            args.Imin = int(sp.stats.norm.ppf(1 - 1e-4, loc=np.mean(spl), scale=np.std(spl)))
            print(f"Selected LambdaA={args.Imin} based on {utils.stat_str(spl, True, prec=5)}")
            if args.Imax is not None and args.Imin >= args.Imax:
                raise Exception(
                    "Mesostable(-1) upper-limit above target order - divergent process or above crit. temp.?"
                )

    if args.Imax is None:
        with utils.timed("sample mesostable for iface B"):
            state1 = state0.copy()
            state1.set_spins(np.ones(state1.n))
            spl = state1.sample_mesostable(progress=args.progress and tee.stderr)
            spl = spl[:, (spl.shape[1] * 2 // 3):]
            args.Imax = int(sp.stats.norm.ppf(1e-10, loc=np.mean(spl), scale=np.std(spl)))
            print(f"Selected LambdaB={args.Imax} based on {utils.stat_str(spl, True, prec=5)}")
            if args.Imin >= args.Imax:
                raise Exception("Mesostable(+1) lower-limit below LambdaA - above crit. temp.?")

    if args.Is is not None:
        ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
        print(f"Interfaces: {np.array(ifs)[:20]} ...")
    else:
        ifs = [args.Imin, args.Imax]
        print(f"Interfaces: dynamic {ifs[0]} .. {ifs[1]}")

    ff = FFSampler([state0], ifs, iface_samples=args.samples)

    try:
        with utils.timed(f"FF compute"):
            ff.compute(progress=args.progress and tee.stderr, timeout=args.timeout, dynamic_ifaces=args.Is is None)

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
                Param='UpSpins' if args.count_spins else 'UpCluster',
                Samples=args.samples,
                N=g.order(),
                M=g.size(),
                Orders=[int(iface.order) for iface in ff.interfaces],
                Clusters=[
                    [int(x) for x in iface.states[0].get_stats().mask] for iface in ff.interfaces
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

    ### Reports

    print(f"FF cising stats:\n{report_runtime_stats()}")

    print()
    print(f"Interface A at {ff.ifaceA.order}, rate 10^{ff.ifaceA.log10_rate:.3f}={10**ff.ifaceA.log10_rate:.5g}")
    print(f"Interface B at {ff.ifaceB.order}, rate 10^{ff.ifaceB.log10_rate:.3f}={10**ff.ifaceB.log10_rate:.5g}")
    nucleus = ff.critical_order_param()
    if nucleus is not None:
        print(f"Critical nucleus size at {nucleus:.5g}")
    print()

    ### Graph drawing

    with utils.timed(f"Ploting {args.fbase + '.html'}"):
        Xs = []
        Es, Es_std = [], []
        ECs, ECs_std = [], []
        UPs = []
        Rates = []

        def apstat(mean, std, vals):
            mean.append(np.mean(vals))
            std.append(np.std(vals))

        for ino, iface in enumerate(ff.interfaces):
            es = []
            for s in iface.states:
                es.append(s.get_hamiltonian())

            Xs.append(iface.order)
            apstat(Es, Es_std, es)
            # apstat(ECs, ECs_std, ces)
            if ino < len(ff.interfaces) - 1:
                UPs.append(iface.up_flow()**(1 / (ff.interfaces[ino + 1].order - iface.order)))
            Rates.append(iface.log10_rate)

        data = [
            go.Scatter(x=Xs, y=UPs, yaxis='y1', name="Up probability [per 1 order]"),
            go.Scatter(x=Xs, y=Rates, yaxis='y2', name="Visit rate (up) [sweeps]"),
            go.Scatter(x=Xs,
                       y=Es,
                       error_y=dict(type='data', array=Es_std, visible=True),
                       yaxis='y3',
                       name="Hamiltonian"),
        ]
        layout = go.Layout(
            yaxis=dict(rangemode='tozero', autorange=True),
            yaxis2=dict(showticklabels=False,
                        overlaying='y',
                        side='left',
                        autorange=True),
            yaxis3=dict(title='E', overlaying='y', side='right', autorange=True),
            title=
            f'FF sampling on {gname}, T={args.T:.3g}, F={args.F:.3g}, {args.require_samples} states/iface, {args.cluster_samples} clustering samples'
        )
        plotly.offline.plot(go.Figure(data=data, layout=layout),
                            filename=args.fbase + '.html',
                            auto_open=False,
                            include_plotlyjs='directory')

    print(f"Log in '{args.fbase + '.log'}'")


main()
