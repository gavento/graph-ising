import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
#tf.autograph.set_verbosity(0, alsologtostdout=True)
import plotly
import plotly.graph_objs as go
import tensorflow.compat.v2 as tf

from graph_ising.forward_flux import DirectIsingClusterFFSampler
from graph_ising.utils import get_device, timed


def main():
    BS = 100  # Batch size
    N = 10    # Grid dim
    RS = 20   # Requered samples
    T = 1.1
    F = 0.1
    #g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    g = nx.grid_2d_graph(N, N)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')

    assert N % 5 == 0
    ifs = list(range(0, N * N, 5))
    print(ifs)
    frs = []  # Fractions of up

    #tf.summary.trace_on(profiler=True)
    # p_drop = exp(-2J/(K_B * T)
    J = 1.0
    p_drop = np.exp(-2 * J / T)
    p_drop = 0.0
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
