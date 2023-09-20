from __future__ import annotations

import argparse
from collections import defaultdict
import math
import time
import timeit

import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns

class DTree:
    def contains(self, point):
        raise NotImplemented

    def depth(self) -> int:
        return 1
    
    def dump(self, space: str) -> None:
        raise NotImplemented
    
    def size(self) -> int:
        return 1

class SingletonTree(DTree):
    def __init__(self, rects):
        self.rects = rects
    def contains(self, point):
        return np.all((point >= self.rects[0]) & (point <= self.rects[1]))
    def dump(self, space):
        print(space, f"singleton {self.rects}")

class ListTree(DTree):
    def __init__(self, rects):
        self.rects = rects
    def contains(self, point):
        return np.any(np.all((point >= self.rects[:, 0]) & (point <= self.rects[:, 1]), axis=1))
    def dump(self, space):
        print(space, f"list ({len(self.rects)}) {self.rects[:5]}...")

class EmptyTree(DTree):
    def contains(self, point):
        return False
    def dump(self, space):
        print(space, "empty")

class FullTree(DTree):
    def contains(self, point):
        return True
    def dump(self, space):
        print(space, "full")

class FulfilledTree(DTree):
    def __init__(self, subtree: DTree, axis: int):
        self.subtree = subtree
        self.axis = axis
    def contains(self, point):
        return self.subtree.contains(without(point, self.axis))
    def dump(self, space):
        print(space, f"fulfilled axis {self.axis} ->")
        self.subtree.dump(space + "\t")
    def depth(self):
        return 1 + self.subtree.depth()
    def size(self):
        return 1 + self.subtree.size()

class SplitDTree(DTree):
    def __init__(self, left: DTree, right: DTree, axis: int, value: float):
        self.left = left
        self.right = right
        self.axis = axis
        self.value = value
    def contains(self, point):
        if point[self.axis] > self.value:
            return self.right.contains(point)
        else:
            return self.left.contains(point)
    def dump(self, space):
        print(space, f"split axis {self.axis} at {self.value}")
        print(space + "  <")
        self.left.dump(space + "\t")
        print(space + "  >")
        self.right.dump(space + "\t")
    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())
    def size(self):
        return 1 + self.left.size() + self.right.size()

class TreeState:
    def __init__(self, dimensions_or_tree, j: int = -1, drop: bool = False):
        if isinstance(dimensions_or_tree, TreeState):
            existing = dimensions_or_tree
            assert(j < existing.dimensions)
            self.depth = existing.depth + 1
            if drop:
                self.dimensions = existing.dimensions - 1
                self.descent = without(existing.descent, j)
            else:
                self.dimensions = existing.dimensions
                self.descent = np.copy(existing.descent)
                self.descent[j] += 1
        else:
            self.depth = 0
            self.dimensions = dimensions_or_tree
            self.descent = np.zeros(dimensions_or_tree)

    def descend(self, j: int) -> TreeState:
        return TreeState(self, j)
    
    def drop(self, j: int) -> TreeState:
        return TreeState(self, j, drop=True)

def without(items, j):
    axis = items.ndim - 1
    indices = [k for k in range(items.shape[axis]) if k != j]
    return np.take(items, indices, axis=axis)

def make_tree_internal(rects, bounds, widths, state: TreeState):
    # print(f"Tree j:{j} bounds:")
    # print(bounds)
    # print("rects")
    # print(f"{rects}")
    #print(" " * state.depth, f"T {len(rects)},{rects.shape[2]}")

    if len(rects) == 0:
        return EmptyTree()
    if len(rects) == 1:
        return SingletonTree(rects[0])
    if len(rects) < 30:
        return ListTree(rects)

    dimensions = rects.shape[2]

    # if all rects completely fill bounds, then fulfilled
    for j in range(dimensions):
        if np.all((rects[:, 0, j] <= bounds[0, j]) & (rects[:, 1, j] >= bounds[1, j])):
            if rects.shape[2] == 1:
                return FullTree()
            else:
                sub_rects = np.unique(without(rects, j), axis=0)
                sub_bounds = without(bounds, j)
                sub_widths = without(widths, j)
                subtree = make_tree_internal(sub_rects, sub_bounds, sub_widths, state.drop(j))
                return FulfilledTree(subtree, j)

    splits = [np.unique(values) for values in (rects[:, :, j] for j in range(dimensions))]
    best_j = 0
    best_classes = 0
    #print(" " * state.depth, "  classes: ", end="")
    for j in range(dimensions):
        #print(f"Dimension {j}")
        sample = len(splits[j])
        sample = 3 if sample > 6 else sample // 2
        #print(f"  splits: {len(splits[j])} ({splits[j][:sample]} ... {splits[j][-sample:]})")
        min = np.min(splits[j])
        max = np.max(splits[j])
        r = max - min
        max_classes = r / widths[j]
        #print(f"  min:{min} max:{max} range:{r} max classes:{max_classes}")
        #print(math.floor(max_classes), end=" ")
        if max_classes > best_classes:
            best_j = j
            best_classes = max_classes
    #print("")

    if best_classes < 1.1: # Diminishing returns as this parameter is dropped; this seems like reasonable trade-off
        # Hardly going to split anything whatever we do, so fall out to a list
        # TODO: This part of the space is almost certainly full, so it would be better to return a FullTree here if possible
        #       (Another doubling of run-time is possible if these are all full)
        # TODO: Just promote the largest rect
        smallest_covered = np.array([np.max(rects[:, 0], axis=0), np.min(rects[:, 1], axis=0)])
        covered_size = np.prod(smallest_covered[1] - smallest_covered[0])
        bounds_size = np.prod(bounds[1] - bounds[0])
        rects_bounds = np.array([np.min(rects[:, 0], axis=0), np.max(rects[:, 1], axis=0)])
        rects_bounds_size = np.prod(rects_bounds[1] - rects_bounds[0])
        covered_fraction = covered_size / rects_bounds_size
        if covered_fraction > 0.1:
            # remainder = rects[np.any(rects[:, 0] < smallest_covered[0], axis=1) | np.any(rects[:, 1] > smallest_covered[1], axis=1)]
            # print(f"fraction: {covered_fraction} rects: {len(rects)} remainder: {len(remainder)} dropped: {len(rects) - len(remainder)}")
            new_rects = np.array([smallest_covered, *rects])
            return ListTree(new_rects)
        else:
            return ListTree(rects)

    j = best_j
    split = splits[j]

    # Split at middle of splits
    if len(split) < 3:
        print(f"WARNING: Can't split as {split} has <3 members, falling back to list")
        return ListTree(rects)
    
    # Try different split positions
    best_split_pos = len(split) // 2
    best_score = len(rects) * 0.6 # If we can't do better than a 75/25 split, might as well just cut down the middle
                                  # with the intuition that might free up other cuts
    lefts = rects[:, 0, j]
    rights = rects[:, 1, j]
    for split_pos in range(len(split)):
        split_at = split[split_pos]
        left_count = (lefts < split_at).sum()
        right_count = (rights > split_at).sum()
        score = left_count if left_count > right_count else right_count
        if best_score is None or score < best_score:
            best_score = score
            best_split_pos = split_pos

    split_at = split[best_split_pos]
    #print(" " * state.depth, f"  - {j} at {split_at}")

    lefts = rects[lefts < split_at] # FIXME: do some thinking about how to handle on the line cases
    lefts[:, 1, j] = np.clip(lefts[:, 1, j], a_max = split_at, a_min = None)
    lefts = np.unique(lefts, axis = 0)
    lefts = lefts[lefts[:, 0, j] < lefts[:, 1, j]] # Filter out empty rectangles

    rights = rects[rights > split_at] # On the line is considered left
    rights[:, 0, j] = np.clip(rights[:, 0, j], a_min = split_at, a_max = None)
    rights = np.unique(rights, axis = 0)
    rights = rights[rights[:, 0, j] < rights[:, 1, j]] # Filter out empty rectangles

    left_bounds = np.copy(bounds)
    left_bounds[1, j] = split_at

    right_bounds = np.copy(bounds)
    right_bounds[0, j] = split_at

    left = make_tree_internal(lefts, left_bounds, widths, state.descend(j))
    right = make_tree_internal(rights, right_bounds, widths, state.descend(j))
    return SplitDTree(left, right, j, split_at)


def make_tree(items, widths):
    axis = items.ndim - 1
    # Reshape each item into a rect
    items = np.unique(items, axis = 0) # No need to look at duplicate items for this process
    rects = np.array([[item - widths, item + widths] for item in items])
    dimensions = items.shape[axis]
    bounds = np.transpose(np.array([[-math.inf, math.inf] for _ in range(dimensions)]))

    return make_tree_internal(rects, bounds, widths, TreeState(dimensions))

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
    
    print("making tree...")
    start = time.time()
    tree = make_tree(sources, np.array([elevation_width, slope_width, access_width]))
    print("build time", time.time() - start)
    print("depth", tree.depth())
    print("size", tree.size())

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

if __name__ == "__main__":
    main()
