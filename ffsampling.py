import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go

from gising import utils
from gising.ff_sampler import CIsingFFSampler
from gising.cising import IsingState


def main():
    parser = utils.default_parser()
    parser.add_argument("--grid", default=10, type=int, help="Use square grid NxN.")
    parser.add_argument("--require_samples",
                        default=10,
                        type=int,
                        help="Samples required at interface.")
    parser.add_argument("--Is", default=10, type=int, help="Interfaces to use.")
    parser.add_argument("--Imin", default=0, type=float, help="Min. interface.")
    parser.add_argument("--Imax", default=None, type=float, help="Max interface.")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
    parser.add_argument("--cluster_samples",
                        default=10,
                        type=int,
                        help="Cluster size samples to run.")
    args = utils.init_experiment(parser)
    assert args.Imax is not None

    g = nx.grid_2d_graph(args.grid, args.grid, periodic=True)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')

    Ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
    print("Interfaces: {}".format(Ifs))

    # p_drop = exp(-2 J / (K_B * T))
    cluster_e_prob = 1.0 - np.exp(-2.0 / args.T)
    if args.cluster_samples == 1:
        cluster_e_prob = 1.0

    state0 = IsingState(graph=g, T=args.T, F=args.F)
    ff = CIsingFFSampler(g,
                         Ifs,
                         state=state0,
                         min_pop_size=args.require_samples,
                         cluster_e_prob=cluster_e_prob,
                         cluster_samples=args.cluster_samples)
    ff.fill_interfaces(progress=True)

    frs = []
    for i, itf in enumerate(ff.interfaces):
        print("Interface", i, itf)
        ups = itf.up_times()
        if len(ups) > 0:
            print("  uptimes   {:6.2f} std {:5.2f}".format(np.mean(ups), np.std(ups)))
        downs = itf.down_times()
        if len(downs) > 0:
            print("  downtimes {:6.2f} std {:5.2f}".format(np.mean(downs), np.std(downs)))
        fr = len(ups) / max(len(ups) + len(downs), 1)
        print("  fraction up", fr)
        frs.append(fr)

    print("FF ran: {} full updates, {} clusterings".format(ff.ran_updates, ff.ran_clusters))

    Xs = []
    Es, Es_std = [], []
    ECs, ECs_std = [], []
    UPs = []

    def apstat(mean, std, vals):
        mean.append(np.mean(vals))
        std.append(np.std(vals))

    with utils.timed("stats"):
        for iface in ff.interfaces:
            es = []
            ces = []
            for p in iface.pops:
                es.append(p.state.get_hamiltonian())
                cs = p.state.mc_max_cluster(samples=args.cluster_samples,
                                            value=1,
                                            edge_prob=cluster_e_prob)
                ces.append(cs.relative_E)

            print("Max Cluster", cs)
            ls = [''.join(".X"[v] for v in r) for r in cs.mask.reshape([args.grid, args.grid])]
            print('\n'.join(ls))

            Xs.append(iface.param)
            apstat(Es, Es_std, es)
            apstat(ECs, ECs_std, ces)
            ups = iface.up_times()
            downs = iface.down_times()
            UPs.append(len(ups) / max(len(ups) + len(downs), 1))

    with utils.timed("plot"):
        data = [
            go.Scatter(x=Xs, y=UPs, yaxis='y', name="Up probability"),
            go.Scatter(x=Xs,
                       y=Es,
                       error_y=dict(type='data', array=Es_std, visible=True),
                       yaxis='y2',
                       name="Total H"),
            go.Scatter(x=Xs,
                       y=ECs,
                       error_y=dict(type='data', array=ECs_std, visible=True),
                       yaxis='y3',
                       name="Cluster H"),
        ]
        layout = go.Layout(
            yaxis=dict(rangemode='tozero', autorange=True),
            yaxis2=dict(title='ETot', overlaying='y', side='right', autorange=True),
            yaxis3=dict(title='EClus', overlaying='y', side='right', autorange=True),
        )
        plotly.offline.plot(go.Figure(data=data, layout=layout),
                            filename=args.fbase + '.html',
                            auto_open=False,
                            include_plotlyjs='directory')

    return

    plt.plot(Ifs, frs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Fraction up')
    plt.xlabel('Interface param')
    title = 'Forward flux sampling, T={:.3g}, F={:.3g}, min. {} samples, {} vertices'.format(
        args.T, args.F, args.require_samples, g.order())
    plt.title(title)
    plt.savefig(args.fbase + ".png", dpi=300)
    plt.show()


main()
