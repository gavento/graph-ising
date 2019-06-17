import argparse
import contextlib
import datetime
import os
import sys
import time

import numpy as np


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", "-c", default=None, type=str, help="Comment for this run.")
    parser.add_argument("--logdir", default="logs", type=str, help="Path to output dir.")
    parser.add_argument("--name", default=None, type=str, help="Main name (default: script name).")
    return parser


def init_experiment(parser):
    args = parser.parse_args()

    if args.name is None:
        args.name = os.path.splitext(sys.argv[0])[0]
    args.full_name = (
        args.name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +
        (('-' + args.comment) if args.comment else ''))
    args.fbase = os.path.join(args.logdir, args.full_name)
    os.makedirs(args.logdir, exist_ok=True)
    args.logfile = open(args.fbase + '.log', 'wt')
    args.logfile.write("Cmd:\n  {}\nArgs:\n{}\n".format(
        ' '.join(sys.argv),
        '\n'.join("  {:19s} {}".format(k, v) for k, v in args.__dict__.items() if k[0] != '_')))
    return args


@contextlib.contextmanager
def timed(name=None, iters=None):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    if iters is not None:
    print((name + " " if name else "") + "took {:.3f} ms".format((t1 - t0)))
