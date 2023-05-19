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
METHOD = str(args['method']).lower()

# Where to save outputs
datapath = os.path.join(os.getcwd(), "data")
try:
    os.mkdir(datapath)
except FileExistsError:
    pass

if METHOD == "permanence":
    add_path = os.path.join(datapath, "add.npy")
    leak_path = os.path.join(datapath, "leak.npy")
    additionality = np.load(add_path)
    leakage = np.load(leak_path)
    c = permanence.net_sequestration(additionality, leakage, 1)
    print(c)
elif METHOD == "additionality":
    additionality = np.array([ 1.0, 1.1 ])
    path = os.path.join(datapath, "add.npy")
    np.save(path, additionality)
elif METHOD == "leakage":
    leakage = np.array([ 0.5, 0.6 ])
    path = os.path.join(datapath, "leak.npy")
    np.save(path, leakage)
else:
    print(f'Unknown methodology {METHOD}')
    sys.exit(-1)
