#!/usr/bin/env python

import argparse
import bz2
import glob
import sys
import json, re

import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import tqdm


def strip_suffix(s):
    s = re.sub('\\.gz$', '', s)
    s = re.sub('\\.bz$', '', s)
    s = re.sub('\\.bz2$', '', s)
    s = re.sub('\\.graphml$', '', s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orca', type=str, default=None, help='Plotly Orca path')
    parser.add_argument('--output',
                        '-o',
                        metavar='OUTBASE',
                        type=str,
                        default=None,
                        help='Output file base (.pdf and .html will be written')
    parser.add_argument('data',
                        metavar='DATA_JSON',
                        type=str,
                        help='GraphML graphs to plot')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.data[:-9]

    with bz2.BZ2File(args.data) as f:
        d = json.load(f)

    hs = np.array([[np.mean(r), np.std(r)] for r in d['Hamiltonians']])

    data = [
        go.Scatter(x=d['Orders'], y=d['UpFlows'], yaxis='y1', name="Probability of reaching next order parameter"),
        go.Scatter(x=d['Orders'], y=d['Log10Rates'], yaxis='y2', name="Transition rate [1/MCSS]"),
        # go.Scatter(x=d['Orders'],
        #            y=hs[:,0],
        #            error_y=dict(type='data', array=hs[:,1], visible=True),
        #            yaxis='y3',
        #            name="Hamiltonian"),
    ]

    layout = go.Layout(
        xaxis=dict(
                domain=[0.0, 1.0]
            ),        
        yaxis=dict(rangemode='tozero', autorange=True, title="P(up)"),
        yaxis2=dict(showticklabels=True, overlaying='y1', side='right', autorange=True,title="Transition rate [log_10]"),
        # yaxis3=dict(title='Hamiltonian', overlaying='y1', side='right', autorange=True),
        title="",
        #f"FFS on {strip_suffix(d['Graph'])}, T={d['T']:.3g}, F={d['F']:.3g}, {d['Samples']} samples/iface",
        legend=dict(x=-.1, y=1.15),
        shapes=[dict(type='line', y0=0, y1=1, line=dict(color='rgb(255, 0, 0)', width=3), x0=d['CSize'], x1=d['CSize'])],
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig,
                        filename=args.output + '.html',
                        auto_open=False,
                        include_plotlyjs='directory')
    print(f"Created {args.output + '.html'}")
    if args.orca:
        plotly.io.orca.config.executable = args.orca
    plotly.io.write_image(fig, args.output + '.pdf')
    print(f"Created {args.output + '.pdf'}")


main()
