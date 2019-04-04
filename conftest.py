import os
import sys

import pytest


#TESTS = os.path.dirname(__file__)
#ROOT = os.path.dirname(TESTS)
#print("ADDING", os.path.join(ROOT, "graph_ising"))
#sys.path.insert(0, os.path.join(ROOT, "graph_ising"))

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
