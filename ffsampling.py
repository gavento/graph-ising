import itertools
import json
import pickle

import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go

from gising import utils
from gising.forward_flux import FFSampler
from gising.ising_state import (ClusterOrderIsingState, SpinCountIsingState,
                                report_runtime_stats)


def main():
    parser = utils.default_parser()
    parser.add_argument("--grid", default=None, type=int, help="Use 2D toroidal grid NxN.")
    parser.add_argument("--grid3d", default=None, type=int, help="Use 3D toroidal grid NxNxN.")
    parser.add_argument("--pref",
                        default=None,
                        type=str,
                        metavar="N,M",
                        help="Use Barabasi-Albert grap on N vertices with M attachments.")
    parser.add_argument("--pcg",
                        default=None,
                        type=str,
                        metavar="N,M,p",
                        help="Use Powerlaw-cluster graph on N vertices with M attachments and p triangle prob.")
    parser.add_argument("--require_samples",
                        "-s",
                        default=10,
                        type=int,
                        help="Samples required at interface.")
    parser.add_argument("--Is", "-I", default=100, type=int, help="Interfaces to use.")
    parser.add_argument("--Imin", default=None, type=float, help="Interface A order param.")
    parser.add_argument("--Imax", default=None, type=float, help="Interface B order param.")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
    parser.add_argument("--timeout", default=100.0, type=float, help="One sim timeout.")
    parser.add_argument("--cluster_samples",
                        "-C",
                        default=None,
                        type=int,
                        help="Cluster size samples to run.")
    parser.add_argument('--count_spins',
                        action='store_const',
                        const=True,
                        default=False,
                        help="Use spin count as the order param (rater than largest cluster size)")
    parser.add_argument(
        "--fix_cluster_seed",
        default=42,
        type=int,
        help="Fix clustering seed (requires high -C for good results), 0 for random.")

    args = utils.init_experiment(parser)
    assert args.Imin is not None
    assert args.Imax is not None

    tee = utils.Tee(args.logfile)
    tee.start()

    with utils.timed("create graph"):
        if args.grid is not None:
            gname = f"2D toroid grid {args.grid}x{args.grid}"
            g = nx.grid_2d_graph(args.grid, args.grid, periodic=True)
        elif args.grid3d is not None:
            gname = f"3D toroid grid {args.grid3d}x{args.grid3d}x{args.grid3d}"
            g = nx.generators.lattice.grid_graph([args.grid3d] * 3, periodic=True)
        elif args.pref is not None:
            gname = f"B-A pref. att. graph {args.pref}"
            n, m = args.pref.split(",")
            g = nx.random_graphs.barabasi_albert_graph(int(n), int(m))
        elif args.pcg is not None:
            gname = f"Powerlaw-cluster graph {args.pcg}"
            n, m, p = args.pcg.split(",")
            g = nx.random_graphs.powerlaw_cluster_graph(int(n), int(m), float(p))
        else:
            raise Exception("Graph type required")
        print(
            f"Created graph with {g.order()} nodes, {g.size()} edges, degrees {utils.stat_str([g.degree(v) for v in g.nodes], True)}"
        )

    ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
    print("Interfaces: {}".format(ifs))

    # cluster_e_prob = 1.0 - np.exp(-2.0 / args.T)
    # cluster_samples = args.cluster_samples
    # if args.cluster_samples is None:
    #     cluster_samples = 1
    #     cluster_e_prob = 1.0

    with utils.timed("create state"):
        if args.count_spins:
            state0 = SpinCountIsingState(g, T=args.T, F=args.F)
        else:
            raise NotImplementedError()
            # TODO: Special clustering options will go here
            state0 = ClusterOrderIsingState(g, T=args.T, F=args.F)

    ff = FFSampler([state0], ifs, iface_samples=args.require_samples)

    try:
        ff.compute(progress=tee.stderr, timeout=args.timeout)
        d = dict(
            name=args.full_name,
            LambdaA=int(ff.interfaces[0].order),
            RateA=ff.interfaces[0].rate,
            LambdaB=int(ff.interfaces[-1].order),
            RateB=ff.interfaces[-1].rate,
            CSize=ff.critical_order_param(),
            T=args.T,
            F=args.F,
            GName=gname,
            Param='UpSpins' if args.count_spins else 'UpCluster',
            Samples=args.require_samples,
            N=g.order(),
            M=g.size(),
            p=float(p) if args.pcg is not None else -1,

            Edges=list(g.edges()),
            Orders=[int(iface.order) for iface in ff.interfaces],
            Clusters=[[[int(x) for x in s.get_stats().mask] for s in iface.states] for iface in ff.interfaces],
            Rates=[iface.rate for iface in ff.interfaces],
            )
        with open(args.fbase+'.json', 'wt') as jf:
            json.dump(d, jf)
    except KeyboardInterrupt:
        print("\nInterrupted, trying to still report anything already computed ...")

    print(f"FF cising stats:\n{report_runtime_stats()}\n")

    print(f"Interface A at {ff.interfaces[0].order}, rate {ff.interfaces[0].rate:.5g}")
    print(f"Interface B at {ff.interfaces[-1].order}, rate {ff.interfaces[-1].rate:.5g}")
    nucleus = ff.critical_order_param()
    if nucleus is not None:
        print(f"Critical nucleus size at {nucleus:.5g}")

    with utils.timed(f"write '{args.fbase + '.ffs.pickle'}'"):
        with open(args.fbase + '.ffs.pickle', 'wb') as f:
            pickle.dump(ff, f)

    Xs = []
    Es, Es_std = [], []
    ECs, ECs_std = [], []
    UPs = []
    Rates = []

    def apstat(mean, std, vals):
        mean.append(np.mean(vals))
        std.append(np.std(vals))

    with utils.timed("stats"):
        for ino, iface in enumerate(ff.interfaces):
            es = []
            for s in iface.states:
                es.append(s.get_hamiltonian())

            Xs.append(iface.order)
            apstat(Es, Es_std, es)
            # apstat(ECs, ECs_std, ces)
            if ino < len(ff.interfaces) - 1:
                UPs.append(iface.up_flow()**(1 / (ff.interfaces[ino + 1].order - iface.order)))
            Rates.append(iface.rate)

    with utils.timed("plot"):
        data = [
            go.Scatter(x=Xs, y=UPs, yaxis='y1', name="Up probability [per 1 order]"),
            go.Scatter(x=Xs, y=Rates, yaxis='y2', name="Visit rate (up) [sweeps]"),
            go.Scatter(x=Xs,
                       y=Es,
                       error_y=dict(type='data', array=Es_std, visible=True),
                       yaxis='y3',
                       name="Hamiltonian"),
            # go.Scatter(x=Xs,
            #            y=ECs,
            #            error_y=dict(type='data', array=ECs_std, visible=True),
            #            yaxis='y3',
            #            name="Cluster H"),
        ]
        layout = go.Layout(
            yaxis=dict(rangemode='tozero', autorange=True),
            yaxis2=dict(showticklabels=False,
                        overlaying='y',
                        side='left',
                        exponentformat='e',
                        type='log',
                        autorange=True),
            yaxis3=dict(title='E', overlaying='y', side='right', autorange=True),
            title=
            f'FF sampling on {gname}, T={args.T:.3g}, F={args.F:.3g}, {args.require_samples} states/iface, {args.cluster_samples} clustering samples'
        )
        plotly.offline.plot(go.Figure(data=data, layout=layout),
                            filename=args.fbase + '.html',
                            auto_open=False,
                            include_plotlyjs='directory')
        print(f"Wrote {args.fbase + '.html'}")


main()
