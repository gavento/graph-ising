import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go

from graph_ising import utils
from graph_ising.ff_sampler import DirectIsingClusterFFSampler


def main():
    parser = utils.default_parser()
    parser.add_argument("--grid", default=10, type=int, help="Use square grid NxN.")
    parser.add_argument("--require_samples", default=10, type=int, help="Samples required at interface.")
    parser.add_argument("--Is", default=10, type=int, help="Interfaces to use.")
    parser.add_argument("--Imin", default=0, type=float, help="Min. interface.")
    parser.add_argument("--IMax", default=None, type=float, help="Max interface.")
    parser.add_argument("-T", default=1.0, type=float, help="Temperature.")
    parser.add_argument("-F", default=0.0, type=float, help="Field.")
    args = utils.init_experiment(parser)
    assert args.Imax is not None

    g = nx.grid_2d_graph(args.grid, args.grid)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')

    Ifs = sorted(set(np.linspace(args.Imin, args.Imax, args.Is, dtype=int)))
    print("Interfaces: {}".format(ifs))

    # p_drop = exp(-2 J / (K_B * T))
    drop_prob = np.exp(-2.0 / args.T)
    drop_samples = 3

    ff = DirectIsingClusterFFSampler(g, ifs, update_fraction=0.1, batch_size=BS, T=T, F=F, drop_edges=p_drop, drop_samples=1)
    for i, itf in enumerate(ff.interfaces):
        while ((len(itf.up_times()) < RS and i < len(ff.interfaces) - 1)):# or
               #(len(itf.down_times()) < RS and i > 0)):
            t = itf.sim_time()
            with timed('adaptive batch from {}, time {:.1f}'.format(itf, t)):
                ff.run_adaptive_batch(itf)
            print("        afterwards:", itf)
        print("Done with", itf)

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

    print("FF ran: {} updates, {} sum update fraction, {} clusterings (counted for in indiv. graphs)".format(ff.ran_updates, ff.ran_updates_frac, ff.ran_clusters))


#    sw = tf.summary.create_file_writer('ising-log')
#    with sw.as_default():
#        tf.summary.trace_export('ising', 1, profiler_outdir='ising-prof')

    plt.plot(ifs, frs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Fraction up')
    plt.xlabel('Interface param')
    title = 'Forward flux sampling, T={:.3g}, F={:.3g}, min. {} samples, {} vertices'.format(
        T, F, RS, g.order())
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()


main()
