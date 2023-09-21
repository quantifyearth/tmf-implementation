import argparse
from collections import defaultdict
import math
import timeit

import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns

class DRangeTree:
    def contains(self, point):
        raise NotImplemented

    def depth(self):
        return 1

class SingletonTree(DRangeTree):
    def __init__(self, source, width):
        if len(source) != len(width):
            raise ValueError("Source had different size to width")
        self._left = source - width
        self._right = source + width
    
    def contains(self, point):
        return np.all([np.greater_equal(point, self._left), np.less_equal(point, self._right)])

    def dump(self, space):
        print(space, f"singleton between {self._left} and {self._right}")

class SingleValueTree(DRangeTree):
    def __init__(self, rest, j, value, width):
        self._rest = rest
        self._j = j
        self._left = value - width
        self._right = value + width
    
    def contains(self, point):
        return point[self._j] > self._left and point[self._j] < self._right and self._rest.contains(without(point, self._j))

    def dump(self, space):
        print(space, f"[{self._j}] in range {self._left} to {self._right} -> (dropping {self._j})")
        self._rest.dump(space + "\t")

    def depth(self):
        return 1 + self._rest.depth()

class AlwaysMatchTree(DRangeTree):
    def contains(self, point):
        return True

    def dump(self, space):
        print(space, f"ALWAYS")

    def depth(self):
        return 0

class FailTree(DRangeTree):
    def contains(self, point):
        return False

    def dump(self, space):
        print(space, f"NEVER")

    def depth(self):
        return 0

class SplitTree(DRangeTree):
    def __init__(self, lefts, rights, j, split):
        self._lefts = lefts
        self._rights = rights
        self._j = j
        self._split = split
    
    def contains(self, point):
        if point[self._j] <= self._split:
            return self._lefts.contains(point)
        else:
            return self._rights.contains(point)

    def dump(self, space):
        print(space, f"[{self._j}] <= {self._split} ->")
        self._lefts.dump(space + "\t")
        print(space, f"[{self._j}] > {self._split} ->")
        self._rights.dump(space + "\t")

    def depth(self):
        return 1 + max(self._lefts.depth(), self._rights.depth())

class ManualTree(DRangeTree):
    def __init__(self, lefts, rights, j, split, width):
        self._lefts = lefts
        self._rights = rights
        self._j = j
        self._split = split
        self._width = width
    
    def contains(self, point):
        if point[self._j] - self._width <= self._split:
            if self._lefts.contains(point):
                return True

        if point[self._j] + self._width >= self._split:
            if self._rights.contains(point):
                return True
        
        return False

    def dump(self, space):
        print(space, f"either [{self._j}] <= {self._split + self._width} ->")
        self._lefts.dump(space + "\t")
        print(space, f"    or [{self._j}] >= {self._split - self._width} ->")
        self._rights.dump(space + "\t")

    def depth(self):
        return 1 + max(self._lefts.depth(), self._rights.depth())

class MiddleFulfilledTree(DRangeTree):
    def __init__(self, lefts, rights, j, left, right, center):
        self._lefts = lefts
        self._rights = rights
        self._j = j
        self._left = left
        self._right = right
        self._center = center
    
    def contains(self, point):
        if point[self._j] < self._left:
            return self._lefts.contains(point)
        elif point[self._j] > self._right:
            return self._rights.contains(point)
        else:
            # Cut out j from point
            p = without(point, self._j)
            return self._center.contains(p)

    def dump(self, space):
        print(space, f"[{self._j}] < {self._left} ->")
        self._lefts.dump(space + "\t")
        print(space, f"[{self._j}] > {self._right} ->")
        self._rights.dump(space + "\t")
        print(space, f"[{self._j}] >= {self._left} and <= {self._right} -> (dropping {self._j})")
        self._center.dump(space + "\t")

    def depth(self):
        return 1 + max(self._lefts.depth(), self._rights.depth(), self._center.depth())

def without(items, j):
    axis = items.ndim - 1
    indices = [k for k in range(items.shape[axis]) if k != j]
    return np.take(items, indices, axis=axis)

# TODO: check when a single axis covers a continuous range with overlaps and then return that large range as a bigger singleton
def make_tree(sources, widths):
    if len(sources) == 0:
        return FailTree()
    if len(sources) == 1:
        return SingletonTree(sources[0], widths)
    # For each vertical axis of sources, sort, find median
    axes = sources.shape[1]
    unique_sorteds = [np.unique(np.sort(sources[:, column])) for column in range(axes)]
    medians = [np.median(unique_sorteds[column]) for column in range(axes)]

    def measure_overlap(width_fraction):
        overlaps = np.zeros(axes)
        for j in range(axes):
            for row in sources:
                if row[j] > medians[j] - widths[j] * (1 - width_fraction) and row[j] < medians[j] + widths[j] * (1 - width_fraction):
                    overlaps[j] += 1
            overlaps[j] /= len(sources)
        return overlaps

    
    limit = len(sources) * 0.75

    # Work out overlap for each axis
    # As a rule of thumb, we're going to make the overlap region half of width
    width_fraction = 0.5
    overlaps = measure_overlap(width_fraction)

    j = np.argmax(overlaps)
    if overlaps[j] > 0.25: # Chance to eat a quarter of values immediately
        # Check for equal values
        if len(np.unique(sources[:, j])) == 1:
            if axes == 1:
                rest = AlwaysMatchTree()
            else:
                rest = make_tree(without(sources, j), without(widths, j))
            return SingleValueTree(rest, j, medians[j], widths[j])
        # Make a three-part split
        left = medians[j] - widths[j] * width_fraction
        right = medians[j] + widths[j] * width_fraction
        left_items = sources[sources[:, j] < left + widths[j]]
        right_items = sources[sources[:, j] > right - widths[j]]
        center_items = sources[(sources[:, j] >= left) & (sources[:, j] <= right)]

        print(f"Centre of {len(sources)} items, axis:{j} left:{left} right:{right} lefts:{len(left_items)} rights:{len(right_items)} centers:{len(center_items)}")
        print(sources)

        if len(left_items) < limit and len(right_items) < limit:
            print("  Making MiddleFulfilledTree")
            lefts = make_tree(left_items, widths)
            rights = make_tree(right_items, widths)
            # Only rows that completely overlap the center band are included
            if axes == 1:
                # Only one axis, so items inside have matched
                center = AlwaysMatchTree()
            else:
                center = make_tree(without(center_items, j), without(widths, j))
            return MiddleFulfilledTree(lefts, rights, j, left, right, center)
        else:
            print("   - Skipping, too unbalanced")
    
    # Find axis with least overlap
    overlaps = measure_overlap(0)
    j = np.argmin(overlaps)
    left_items = sources[sources[:, j] <= medians[j] + widths[j]]
    right_items = sources[sources[:, j] > medians[j] - widths[j]]
    print(f"Split of {len(sources)} items, lefts:{len(left_items)} rights:{len(right_items)}")

    if len(left_items) < limit and len(right_items) < limit:
        print("  Making SplitTree")
        lefts = make_tree(left_items, widths)
        rights = make_tree(right_items, widths)
        return SplitTree(lefts, rights, j, medians[j])
    else:
        print("    - Making manual k-d tree as too unbalanced")
        left_items = sources[sources[:, j] <= medians[j]]
        right_items = sources[sources[:, j] > medians[j]]
        print(f"Manual of {len(sources)} items, j:{j} median:{medians[j]} lefts:{len(left_items)} rights:{len(right_items)}")
        lefts = make_tree(left_items, widths)
        rights = make_tree(right_items, widths)
        return ManualTree(lefts, rights, j, medians[j], widths[j])


def k_search_test(
    k_filename: str,
    start_year: int,
) -> None:

    luc0, luc5, luc10 = luc_matching_columns(start_year)

    source_pixels = pd.read_parquet(k_filename)
    # Split source_pixels into classes
    source_classes = defaultdict(list)
    elevation_range = [math.inf, -math.inf]
    slope_range = [math.inf, -math.inf]
    access_range = [math.inf, -math.inf]
    elevation_width = 200
    slope_width = 2.5
    access_width = 10
    for _, row in source_pixels.iterrows():
        key = (int(row.ecoregion) << 16) | (int(row[luc0]) << 10) | (int(row[luc5]) << 5) | (int(row[luc10]))
        if key != 1967137: continue
        source_classes[key].append(row)
        if row.elevation - elevation_width < elevation_range[0]: elevation_range[0] = row.elevation - elevation_width
        if row.elevation + elevation_width > elevation_range[1]: elevation_range[1] = row.elevation + elevation_width

        if row.slope - slope_width < slope_range[0]: slope_range[0] = row.slope - slope_width
        if row.slope + slope_width > slope_range[1]: slope_range[1] = row.slope + slope_width

        if row.access - access_width < access_range[0]: access_range[0] = row.access - access_width
        if row.access + access_width > access_range[1]: access_range[1] = row.access + access_width

    source_nps = dict()
    for key, values in source_classes.items():
        source_nps[key] = np.array([(row.elevation, row.slope, row.access) for row in values])
    

    # Invent an array of values
    length = 1000
    elevation = np.random.uniform(elevation_range[0] - elevation_width, elevation_range[1] + elevation_width, (length))
    slopes = np.random.uniform(slope_range[0] - slope_width, slope_range[1] + slope_width, (length))
    access = np.random.uniform(access_range[0] - access_width, access_range[1] + access_width, (length))

    sources = source_nps[1967137]

    def do_np_matching():
        found = 0
        for i in range(length):
            pos = np.where(elevation[i] + 200 < sources[:, 0], False,
                            np.where(elevation[i] - 200 > sources[:, 0], False,
                            np.where(slopes[i] + 2.5 < sources[:, 1], False,
                            np.where(slopes[i] - 2.5 > sources[:, 1], False,
                            np.where(access[i] + 10 < sources[:, 2], False,
                            np.where(access[i] - 10 > sources[:, 2], False,
                                        True
                        ))))))
            found += 1 if np.any(pos) else 0
        return found
    
    tree = make_tree(sources, np.array([elevation_width, slope_width, access_width]))
    print("depth", tree.depth())

    def do_nd_tree_matching():
        found = 0
        for i in range(length):
            found += 1 if tree.contains(np.array([elevation[i], slopes[i], access[i]])) else 0
        return found

    def speed_of(what, func):
        t = timeit.Timer(stmt=func)
        loops, time = t.autorange()
        print(what, ": ", time / loops, "per call")
    speed_of("NP matching", do_np_matching)
    speed_of("Tree matching", do_nd_tree_matching)



def main():
    parser = argparse.ArgumentParser(description="Finds all potential matches to K in matching zone, aka set S.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_filename",
        help="Parquet file containing pixels from K as generated by calculate_k.py"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file desribing area from which matching pixels may be selected."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        dest="start_year",
        help="Year project started."
    )
    parser.add_argument(
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Year of project evalation"
    )
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--cpc",
        type=str,
        required=True,
        dest="cpc_directory_path",
        help="Filder containing Coarsened Proportional Coverage GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        dest="ecoregions_directory_path",
        help="Directory containing Ecoregions GeoTIFF tiles."
    )
    parser.add_argument(
        "--elevation",
        type=str,
        required=True,
        dest="elevation_directory_path",
        help="Directory containing SRTM elevation GeoTIFF tiles."
    )
    parser.add_argument(
        "--slope",
        type=str,
        required=True,
        dest="slope_directory_path",
        help="Directory containing slope GeoTIFF tiles."
    )
    parser.add_argument(
        "--access",
        type=str,
        required=True,
        dest="access_directory_path",
        help="Directory containing access to health care GeoTIFF tiles."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination parquet file for results."
    )

    args = parser.parse_args()

    k_search_test(
        args.k_filename,
        args.start_year,
    )

if __name__ == "__main__":
    main()

def test():
    array = np.array([
        [1, 3],
        [1, 6],
        [1, 9],
        [2, 9],
        [10, 9],
        [11, 9],
    ])
    widths = np.array([1, 2])
    tree = make_tree(array, widths)
    print(array)
    print(tree)
    tree.dump("")
    for a in array:
        print(tree.contains(a))
        print(tree.contains(a - (widths * 0.9)))
        print(tree.contains(a + (widths * 0.9)))
        print(tree.contains(a - (widths * 1.1)))

