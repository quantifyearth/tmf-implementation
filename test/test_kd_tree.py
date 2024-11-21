import math
from time import time
import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns
from methods.utils.kd_tree import KDRangeTree, make_kdrangetree, make_rumba_tree

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

def test_kd_tree_matches_as_expected():
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

    def build_kdranged_tree_for_k(k_rows) -> KDRangeTree:
        return make_kdrangetree(np.array([(
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
            ]), ALLOWED_VARIATION)


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
    kd_tree = build_kdranged_tree_for_k(source_rows)
    print("build time", time() - start)
    print("tree depth", kd_tree.depth())
    print("tree size", kd_tree.size())

    def do_kdrange_tree_matching():
        found = 0
        for i in range(length):
            found += 1 if len(kd_tree.members(test_values[i])) > 0 else 0
        return found

    rumba_tree = make_rumba_tree(kd_tree, source)

    def do_rumba_tree_matching():
        found = 0
        for i in range(length):
            found += 1 if len(rumba_tree.members(test_values[i])) > 0 else 0
        return found

    test_np_matching = False # This is slow but a useful check so I don't want to delete it
    if test_np_matching:
        speed_of("NP matching", do_np_matching)
    speed_of("KD Tree matching", do_kdrange_tree_matching)
    speed_of("Rumba matching", do_rumba_tree_matching)

def test_rumba_tree_sampling():
    """Check that the rumba tree members_sample function returns a uniform random sample.
    
    Actually only tests the mean converges to the middle index over a series of runs.
    """
    data = np.arange(3000).reshape((-1, 3))

    # Build a tree
    centre = np.array([1500, 1500, 1500])

    kd_tree = make_kdrangetree(data, centre)
    rumba_tree = make_rumba_tree(kd_tree, data)

    assert 1000 == rumba_tree.count_members(centre)

    means = []
    for seed in range(100):
        for i in range(100):
            sample = rumba_tree.members_sample(centre, i + 1, np.random.default_rng(100 * seed + i))
            means.append(np.mean(sample))
    mean = np.mean(np.array(means))
    mean_difference = abs(mean - 500)
    assert mean_difference < 1
