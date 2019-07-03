#!/usr/bin/env python

import argparse
import bz2
import glob
import sys

import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import tqdm


def degdist(g, up_to=None):
    if up_to is None:
        up_to = g.order()
    ds = np.zeros(g.order())
    for _, d in g.degree():
        ds[d] += 1
    nonz = 0
    for i, x in enumerate(ds):
        if x > 0:
            nonz = i
    return ds[:min(up_to, nonz)] / g.order()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upto',
                        '-u',
                        type=int,
                        default=None,
                        help='Cutoff for degrees to display (default: all)')
    parser.add_argument('--orca',
                        type=str,
                        default=None,
                        help='Plotly Orca path')
    parser.add_argument('--output', '-o',
                        metavar='OUTBASE',
                        type=str,
                        required=True,
                        help='Output file base (.pdf and .html will be written')
    parser.add_argument('graphs',
                        metavar='GRAPHML',
                        type=str,
                        nargs='*',
                        help='GraphML graphs to plot')
    parser.add_argument('--group',
                        '-g',
                        metavar='PATTERN',
                        type=str, action='append',
                        help='Group of graphs to plot (one glob pattern)')
    args = parser.parse_args()
    data = []

    for fn in args.graphs:
        g = nx.read_graphml(fn)
        ds = degdist(g, args.upto)
        sys.stderr.write(f'Read {fn}\n')
        data.append(
            go.Scatter(y=ds,
                       name=f"{fn} avgdeg={2 * g.size() / g.order():.3f}",
                       line=dict(width=1.5)))

    for fpat in args.group:
        degs = []
        avgdegs = []
        for fn in glob.glob(fpat):
            g = nx.read_graphml(fn)
            degs.append(degdist(g, args.upto))
            avgdegs.append(2 * g.size() / g.order())
            sys.stderr.write(f' Read {fn}\n')
        sys.stderr.write(f'Group {fpat} expanded to {len(degs)} graphs\n')
        l = max(len(d) for d in degs)
        degs2 = [np.pad(d, (0, l - len(d))) for d in degs]
        data.append(
            go.Scatter(x=np.arange(len(degs2[0])),
                       y=np.mean(degs2, axis=0),
                       error_y=dict(array=np.std(degs2, axis=0)),
                       name=f"{fpat} [{len(degs2)}x] avgdeg={np.mean(avgdegs):.3f}",
                       line=dict(width=1.5)))

    layout = go.Layout(yaxis=dict(rangemode='tozero', autorange=True, title='Degree density'),
                       xaxis=dict(title='Degree'),
                       title=None,
                       legend=dict(x=-.1, y=1.2))
    fig = go.Figure(data=data, layout=layout)

    plotly.offline.plot(fig,
                        filename=args.output + '.html',
                        auto_open=False,
                        include_plotlyjs='directory')
    if args.orca:
        plotly.io.orca.config.executable = args.orca
    plotly.io.write_image(fig, args.output + '.pdf')


main()
