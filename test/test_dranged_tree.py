import math
from time import time
import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns
from methods.utils.dranged_tree import DRangedTree

ALLOWED_VARIATION = np.array([
    200,
    2.5,
    10,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
])

def test_dranged_tree_matches_as_expected():
    def build_rects(items):
        rects = []
        for item in items:
            lefts = []
            rights = []
            for dimension, value in enumerate(item):
                width = ALLOWED_VARIATION[dimension]
                if width < 0:
                    fraction = -width
                    width = value * fraction
                lefts.append(value - width)
                rights.append(value + width)
            rects.append([lefts, rights])
        return np.array(rects)

    expected_fraction = 1 / 100 # This proportion of pixels we end up matching

    def build_dranged_tree_for_k(k_rows) -> DRangedTree:
        return DRangedTree.build(np.array([(
                row.elevation,
                row.slope,
                row.access,
                row["cpc0_u"],
                row["cpc0_d"],
                row["cpc5_u"],
                row["cpc5_d"],
                row["cpc10_u"],
                row["cpc10_d"],
                ) for row in k_rows
            ]), ALLOWED_VARIATION, expected_fraction)


    luc0, luc5, luc10 = luc_matching_columns(2012)
    source_pixels = pd.read_parquet("./test/data/1201-k.parquet")

    # Split source_pixels into classes
    source_rows = []
    for _, row in source_pixels.iterrows():
        key = (int(row.ecoregion) << 16) | (int(row[luc0]) << 10) | (int(row[luc5]) << 5) | (int(row[luc10]))
        if key != 1967137:
            continue
        source_rows.append(row)

    source = np.array([
        [
            row.elevation,
            row.slope,
            row.access,
            row["cpc0_u"],
            row["cpc0_d"],
            row["cpc5_u"],
            row["cpc5_d"],
            row["cpc10_u"],
            row["cpc10_d"],
        ] for row in source_rows
    ])

    # Invent an array of values that matches the expected_fraction
    length = 10000
    np.random.seed(42)

    ranges = np.transpose(np.array([
        np.min(source, axis=0),
        np.max(source, axis=0)
    ]))

    # Safe ranges (exclude 10% of outliers)
    safe_ranges = np.transpose(np.array([
        np.quantile(source, 0.05, axis=0),
        np.quantile(source, 0.95, axis=0)
    ]))

    # Need to put an estimate here of how much of the area inside those 90% bounds is actually filled
    filled_fraction = 0.775

    # Proportion of values that should fall inside each dimension
    inside_fraction = expected_fraction * math.pow(1 / filled_fraction, len(ranges))
    inside_length = math.ceil(length * inside_fraction)
    inside_values = np.random.uniform(safe_ranges[:, 0], safe_ranges[:, 1], (inside_length, len(ranges)))

    widths = ranges[:, 1] - ranges[:, 0]
    range_extension = 100 * widths # Width extension makes it very unlikely a random value will be inside
    outside_ranges = np.transpose([ranges[:, 0] - range_extension, ranges[:, 1] + range_extension])

    outside_length = length - inside_length
    outside_values = np.random.uniform(outside_ranges[:, 0], outside_ranges[:, 1], (outside_length, len(ranges)))

    test_values = np.append(inside_values, outside_values, axis=0)

    def do_np_matching():
        source_rects = build_rects(source)
        found = 0
        for i in range(length):
            pos = np.all((test_values[i] >= source_rects[:, 0]) & (test_values[i] <= source_rects[:, 1]), axis=1)
            found += 1 if np.any(pos) else 0
        return found

    def speed_of(what, func):
        expected_finds = 946
        start = time()
        value = func()
        end = time()
        assert value == expected_finds, f"Got wrong value {value} for method {what}, expected {expected_finds}"
        print(what, ": ", (end - start) / length, "per call")

    print("making tree... (this will take a few seconds)")
    start = time()
    tree = build_dranged_tree_for_k(source_rows)
    print("build time", time() - start)
    print("tree depth", tree.depth())
    print("tree size", tree.size())

    def do_drange_tree_matching():
        found = 0
        for i in range(length):
            found += 1 if tree.contains(test_values[i]) else 0
        return found

    test_np_matching = False # This is slow but a useful check so I don't want to delete it
    if test_np_matching:
        speed_of("NP matching", do_np_matching)
    speed_of("Tree matching", do_drange_tree_matching)
