import argparse
import sys
import os
import numpy as np

from methods import permanence
import arkdir

parser = argparse.ArgumentParser(description="TMF Methodology Implementation.")

# TODO: This will be split up most likely but for now,
# in order to test the pipeline it is all going in here.
parser.add_argument(
    '--method',
    help="Which part of the methodology to run",
    required=True,
    dest="method"
)

args = vars(parser.parse_args())
method = str(args['method']).lower()

# Where to save outputs, we could probably do better.
arkpath = arkdir.Arkpath()

if method == "permanence":
    add_path = arkpath.load("add.npy")
    leak_path = arkpath.load("leak.npy")
    additionality = np.load(add_path)
    leakage = np.load(leak_path)
    c = permanence.net_sequestration(additionality, leakage, 1)
    print(c)
elif method == "additionality":
    additionality = np.array([ 1.0, 1.1 ])
    path = arkpath.save("add.npy")
    np.save(path, additionality)
elif method == "leakage":
    leakage = np.array([ 0.5, 0.6 ])
    path = arkpath.save("leak.npy")
    np.save(path, leakage)
else:
    print(f'Unknown methodology {method}')
    sys.exit(-1)

