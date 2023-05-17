import argparse
import sys
import os
import numpy as np

from methods import permanence

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
ark_dir = os.getenv("ARKDIR", default=os.getcwd())

if method == "permanence":
    add_path = os.path.join(ark_dir, "add.npy")
    leak_path = os.path.join(ark_dir, "leak.npy")
    additionality = np.load(add_path)
    leakage = np.load(leak_path)
    c = permanence.net_sequestration(additionality, leakage, 1)
    print(c)
elif method == "additionality":
    additionality = np.array([ 1.0, 1.1 ])
    path = os.path.join(ark_dir, "add")
    np.save(path, additionality)
elif method == "leakage":
    leakage = np.array([ 0.5, 0.6 ])
    path = os.path.join(ark_dir, "leak")
    np.save(path, leakage)
else:
    print(f'Unknown methodology {method}')
    sys.exit(-1)

