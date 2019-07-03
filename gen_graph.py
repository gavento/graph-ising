#!/usr/bin/env python

import argparse
import bz2
import sys

import networkx as nx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        metavar='OUTFILE',
                        type=str,
                        required=True,
                        help='Write to file (compressed if .bz2 or .gz)')
    subparsers = parser.add_subparsers(dest='cmd')

    g_grid = subparsers.add_parser('grid', help='Grid graphs (any dim)')
    g_grid.add_argument('dims',
                        metavar='N',
                        type=int,
                        nargs='+',
                        help='Grid sizes (one for every dimension)')
    g_grid.add_argument('--wrap', '-w', action='store_true', default=False, help="Toroidal grid")

    g_pref = subparsers.add_parser('pref', help='Preferential attachment and Barabasi-Albert')
    g_pref.add_argument('n', metavar='N', type=int, help='Number of vertices')
    g_pref.add_argument('m', metavar='M', type=int, help='Edges for every new vertex')
    g_pref.add_argument('p',
                        metavar='P',
                        type=float,
                        default=0.0,
                        nargs='?',
                        help='Triangle add prob.')

    g_edges = subparsers.add_parser('edges', help='Convert list-of-edges file to graphml')
    g_edges.add_argument('f', metavar='FILE', type=str, help='Input file (#-comments ignored)')
    g_edges.add_argument('--dir',
                         '-d',
                         action='store_true',
                         default=False,
                         help="Make the graph directed")

    args = parser.parse_args()

    if args.cmd == 'grid':
        g = nx.generators.lattice.grid_graph(args.dims, periodic=args.wrap)
    elif args.cmd == 'pref':
        g = nx.random_graphs.powerlaw_cluster_graph(args.n, args.m, args.p)
    elif args.cmd == 'edges':
        es = []
        with open(args.f) as f:
            for l in f:
                if l.find('#') >= 0:
                    l = l[:l.find('#')]
                es.extend(l.split())
        assert (len(es) % 2 == 0)
        edges = list(zip(es[0::2], es[1::2]))
        if args.dir:
            g = nx.DiGraph(edges)
        else:
            g = nx.Graph(edges)

    nx.write_graphml(g, args.o)

    sys.stderr.write(
        f'Created graph "{args.o}" with |V|={g.order()}, |E|={g.size()}, avgdeg={2 * g.size() / g.order():.5g}\n'
    )


if __name__ == '__main__':
    main()
