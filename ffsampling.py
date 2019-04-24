import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf

from graph_ising.forward_flux import DirectIsingClusterFFSampler
from graph_ising.utils import get_device, timed

tf.autograph.set_verbosity(0, alsologtostdout=True)


def main():
    BS = 100  # Batch size
    N = 10  # Grid dim
    RS = 50  # Requered samples
    T = 1.3
    F = 0.0
    #g = nx.random_graphs.powerlaw_cluster_graph(N, 3, 0.5)
    g = nx.grid_2d_graph(N, N)
    g = nx.relabel.convert_node_labels_to_integers(g, ordering='sorted')

    assert N % 10 == 0
    ifs = list(range(0, N, 2)) + list(range(N, N * N - N, 5)) + list(range(N * N - N, N * N, 2))
    print(ifs)
    frs = []  # Fractions of up

    tf.summary.trace_on(profiler=True)

    ff = DirectIsingClusterFFSampler(g, ifs, update_fraction=0.5, batch_size=BS, T=T, F=F)
    for i, itf in enumerate(ff.interfaces):
        while ((len(itf.up_times()) < RS and i < len(ff.interfaces) - 1) or
               (len(itf.down_times()) < RS and i > 0)):
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
        fr = len(ups) / (len(ups) + len(downs))
        print("  fraction up", fr)
        frs.append(fr)

    sw = tf.summary.create_file_writer('ising-log')
    with sw.as_default():
        tf.summary.trace_export('ising', 1, profiler_outdir='ising-prof')

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
