import itertools
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go

from gising import utils
from gising.cising import IsingState, report_stats
from gising.ff_sampler import CIsingFFSampler


def main():
    parser = utils.default_parser()
    parser.add_argument("--grid", default=None, type=int, help="Use 2D toroidal grid NxN.")
    parser.add_argument("--grid3d", default=None, type=int, help="Use 3D toroidal grid NxNxN.")
    parser.add_argument("--pref",
                        default=None,
                        type=str,
                        metavar="N,M",
                        help="Use Barabasi-Albert grap on N vertices with M attachments.")
    parser.add_argument("--require_samples",
                        "-s",
                        default=10,
                        type=int,
                        help="Samples required at interface.")
    parser.add_argument("--Is", default=10, type=int, help="Interfaces to use.")
    parser.add_argument("--Imin", default=0, type=float, help="Min. interface.")
    parser.add_argument("--Imax", default=None, type=float, help="Max interface.")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
    parser.add_argument("--timeout", default=100.0, type=float, help="One sim timeout.")
    parser.add_argument("--cluster_samples",
                        "-C",
                        default=None,
                        type=int,
                        help="Cluster size samples to run.")
    args = utils.init_experiment(parser)
    assert args.Imax is not None

    tee = utils.Tee(args.logfile)
    tee.start()

    with utils.timed("create graph"):
        if args.grid is not None:
            gname = f"2D toroid grid {args.grid}x{args.grid}"
            g = nx.grid_2d_graph(args.grid, args.grid, periodic=True)
            g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
        elif args.grid3d is not None:
            gname = f"3D toroid grid {args.grid3d}x{args.grid3d}x{args.grid3d}"
            g = nx.generators.lattice.grid_graph([args.grid3d] * 3, periodic=True)
            g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')
        elif args.pref is not None:
            gname = f"Bar.-Alb. pref. att. graph {args.pref}"
            n, m = args.pref.split(",")
            g = nx.random_graphs.barabasi_albert_graph(int(n), int(m))
        else:
            raise Exception("Graph type required")
        print(
            f"Created graph with {g.order()} nodes, {g.size()} edges, degrees {utils.stat_str([g.degree(v) for v in g.nodes])}"
        )

    Ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
    print("Interfaces: {}".format(Ifs))

    # p_drop = exp(-2 J / (K_B * T))
    cluster_e_prob = 1.0 - np.exp(-2.0 / args.T)
    cluster_samples = args.cluster_samples
    if args.cluster_samples is None:
        cluster_samples = 1
        cluster_e_prob = 1.0

    with utils.timed("create state"):
        state0 = IsingState(graph=g, T=args.T, F=args.F)

    ff = CIsingFFSampler(g,
                         Ifs,
                         state=state0,
                         min_pop_size=args.require_samples,
                         cluster_e_prob=cluster_e_prob,
                         cluster_samples=cluster_samples)

    ff.fill_interfaces(progress=tee.stderr, timeout=args.timeout)
    print(f"FF cising stats:\n{report_stats()}")

    with utils.timed(f"write '{args.fbase + '-FF.pickle'}'"):
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
            ces = []
            for p in iface.pops:
                es.append(p.state.get_hamiltonian())
                ces.append(p.cluster_stats.relative_E)

            #print("Max Cluster", cs)
            #ls = [''.join(".X"[v] for v in r) for r in cs.mask.reshape([args.grid, args.grid])]
            #print('\n'.join(ls))

            Xs.append(iface.param)
            apstat(Es, Es_std, es)
            apstat(ECs, ECs_std, ces)
            if ino < len(ff.interfaces) - 1:
                UPs.append(iface.normalized_upflow(ff.interfaces[ino + 1].param - iface.param))
            Rates.append(iface.rate)
            # TODO: use iface.rate, check eq.?
            #if ino == 0:
            #    Rates.append(1.0 / np.mean(ff.ifaceA_up_up_times))
            #else:
            #    Rates.append(Rates[-1] * ff.interfaces[ino - 1].normalized_upflow(1.0))

    with utils.timed("plot"):
        data = [
            go.Scatter(x=Xs, y=UPs, yaxis='y1', name="Up probability (normed to param)"),
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
            yaxis2=dict(
                showticklabels=False,
                overlaying='y',
                side='left',
                #anchor='free',
                #position=0.05,
                exponentformat='e',
                #tickformat='~.3g',
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
