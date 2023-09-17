import argparse
from collections import defaultdict
import math
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
        print(space, f"list {self.rects}")

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

def without(items, j):
    axis = items.ndim - 1
    indices = [k for k in range(items.shape[axis]) if k != j]
    return np.take(items, indices, axis=axis)

def make_tree_internal(rects, bounds, j: int, depth: int):
    # print(f"Tree j:{j} bounds:")
    # print(bounds)
    # print("rects")
    # print(f"{rects}")
    #print(" " * depth, f"T {len(rects)},{rects.shape[2]}")

    if len(rects) == 0:
        return EmptyTree()
    if len(rects) == 1:
        return SingletonTree(rects[0])
    if len(rects) < 30:
        return ListTree(rects)

    # if all rects completely fill bounds, then fulfilled
    if np.all((rects[:, 0, j] <= bounds[0, j]) & (rects[:, 1, j] >= bounds[1, j])):
        if rects.shape[2] == 1:
            return FullTree()
        else:
            sub_rects = without(rects, j)
            sub_bounds = without(bounds, j)
            next_j = j % sub_rects.shape[2]
            subtree = make_tree_internal(sub_rects, sub_bounds, next_j, depth + 1)
            return FulfilledTree(subtree, j)
        
    # TODO: if min and max of left sides are equal or min and max of right sides, put a cut there
    # TODO: if min of left and max of right are equal, single value so need a new tree node that checks that value
    #       then drops the dimension
    
    # Doesn't matter which axis so just cycle through
    # TODO: find a median that is aligned with *something*, so in case of even number (always even due to left/right),
    #        pick point nearest mean
    # TODO: pick j based on minimising overlap as before
    # TODO: track how many splits in each dimension and after N (maybe 2 or 3?) do an upper or lower bound
    #        if a bound is still infinite for earlier out when searching (else OOB values need to get to furthest
    #        leaf every time)
    # TODO: clip rects by bounds here so we don't need to clip below
    # print(f"values {rects[:, :, j]}")
    centers = np.mean(rects[:, :, j], axis=1)
    widths = rects[:, 1, j] - rects[:, 0, j]
    total_widths = np.sum(widths)

    weighted_center = np.sum(centers * widths) / total_widths

    # Find point in rects[:, :, j] closest to weighted_center in direction of FIXME:???
    # We want to find a specific point to make the split as efficient as possible
    sorted_points = np.sort(rects[:, :, j], axis=None)

    min = sorted_points[0]
    max = sorted_points[-1]
    sorted_points = sorted_points[(sorted_points > min) & (sorted_points < max)]
    if len(sorted_points > 0):
        # print(f"sorted_points {sorted_points}")
        side = "left"
        chosen = np.searchsorted(sorted_points, weighted_center, side=side)
        if chosen == len(sorted_points):
            chosen -= 1
        split_at = sorted_points[chosen]
        # print(f"side {side} chosen {chosen} final median {median}")
    else:
        # Optimise: all points cover all area, so we just need a bounds check here and we're golden
        bound = bounds[:, j]
        next_j = (j + 1) % rects.shape[2]
        if bound[0] < min:
            new_bounds = np.copy(bounds)
            new_bounds[0, j] = min
            rest = make_tree_internal(rects, new_bounds, next_j, depth + 1)
            return SplitDTree(EmptyTree(), rest, j, min)
        elif bound[1] > max:
            new_bounds = np.copy(bounds)
            new_bounds[1, j] = max
            rest = make_tree_internal(rects, new_bounds, next_j, depth + 1)
            return SplitDTree(rest, EmptyTree(), j, max)
        else:
            raise RuntimeError(f"j: {j} min: {min} max: {max} bounds: {bounds} now what?")
    
    if False:
        # print(f"centers {centers}")
        median = np.median(centers)
        # print(f"median {median}")
        mean = np.mean(bounds[:, j])
        if math.isnan(mean) or math.isinf(mean):
            # Bounds not bounded, so use average of centers instead
            mean = np.mean(centers)
        # print(f"mean {mean}")
        # Find point in rects[:, :, j] closest to median in direction of mean
        # We want to find a specific point to make the split as efficient as possibles
        sorted_points = np.sort(rects[:, :, j], axis=None)

        mean = np.median(sorted_points) # Not really a mean, more of the target point
        # We cannot use the first or last value to split, as that's not a split
        min = sorted_points[0]
        max = sorted_points[-1]
        sorted_points = sorted_points[(sorted_points > min) & (sorted_points < max)]
        if len(sorted_points > 0):
            # print(f"sorted_points {sorted_points}")
            side = "right" if mean < median else "left"
            chosen = np.searchsorted(sorted_points, median, side=side)
            if side == "right" or chosen == len(sorted_points):
                chosen -= 1
            median = sorted_points[chosen]
            # print(f"side {side} chosen {chosen} final median {median}")
        else:
            # Optimise: all points cover all area, so we just need a bounds check here and we're golden
            bound = bounds[:, j]
            next_j = (j + 1) % rects.shape[2]
            if bound[0] < min:
                new_bounds = np.copy(bounds)
                new_bounds[0, j] = min
                rest = make_tree_internal(rects, new_bounds, next_j, depth + 1)
                return SplitDTree(EmptyTree(), rest, j, min)
            elif bound[1] > max:
                new_bounds = np.copy(bounds)
                new_bounds[1, j] = max
                rest = make_tree_internal(rects, new_bounds, next_j, depth + 1)
                return SplitDTree(rest, EmptyTree(), j, max)
            else:
                raise RuntimeError(f"j: {j} min: {min} max: {max} bounds: {bounds} now what?")

    lefts = rects[rects[:, 0, j] < split_at] # FIXME: do some thinking about how to handle on the line cases
    lefts[:, 1, j] = np.clip(lefts[:, 1, j], a_max = split_at, a_min = None)
    lefts = np.unique(lefts, axis = 0)
    #lefts = lefts[lefts[:, 0, j] < lefts[:, 1, j]] # Filter out empty rectangles

    # print("lefts")
    # print(lefts)

    rights = rects[rects[:, 1, j] > split_at] # On the line is considered left
    rights[:, 0, j] = np.clip(rights[:, 0, j], a_min = split_at, a_max = None)
    rights = np.unique(rights, axis = 0)
    #rights = rights[rights[:, 0, j] < rights[:, 1, j]] # Filter out empty rectangles

    # print("rights")
    # print(rights)

    left_bounds = np.copy(bounds)
    left_bounds[1, j] = split_at

    right_bounds = np.copy(bounds)
    right_bounds[0, j] = split_at

    next_j = (j + 1) % rects.shape[2]
    left = make_tree_internal(lefts, left_bounds, next_j, depth + 1)
    right = make_tree_internal(rights, right_bounds, next_j, depth + 1)
    return SplitDTree(left, right, j, split_at)


def make_tree(items, widths):
    axis = items.ndim - 1
    # Reshape each item into a rect
    items = np.unique(items, axis = 0) # No need to look at duplicate items for this process
    rects = np.array([[item - widths, item + widths] for item in items])
    bounds = np.transpose(np.array([[-math.inf, math.inf] for _ in range(items.shape[axis])]))
    # Call internal with the rects and the bounds
    return make_tree_internal(rects, bounds, 0, 0)

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
    
    print("making tree...")
    tree = make_tree(sources, np.array([elevation_width, slope_width, access_width]))
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
