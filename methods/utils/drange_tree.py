"""
This file and associated classes represent a d-range-tree, which is a name I made up.
The idea of a d-range tree is you can build it from a list of N-dimensional ranges,
and then query points for membership.

Currently all the ranges are the same size but you could easily build it with more varied ranges.
Given all ranges are the same size, you might be wondering "why not just do a range query on a k-d tree?"
The answer is this is faster, at least for our data sets - the intuition being, our ranges overlap a lot,
so a structure that merges those ranges together will be faster than a k-d tree which knows nothing about
the overlapped structure.
"""

from __future__ import annotations

import numpy as np
import math

# TODO: Implement without numpy for actual matching
# TODO: Try an implementation with a BSP so we can actually merge ranges together.

class DRangeTree:
    def contains(self, point):
        raise NotImplemented

    def depth(self) -> int:
        return 1
    
    def dump(self, space: str) -> None:
        raise NotImplemented
    
    def size(self) -> int:
        return 1
    
    @staticmethod
    def build(items, widths) -> DRangeTree:
        items = np.unique(items, axis = 0) # Ditch duplicate points
        # Reshape each point into a hyper-rect +- width
        rects = np.array([[item - widths, item + widths] for item in items])
        # Calculate initial bounds
        dimensions = items.shape[items.ndim - 1]
        bounds = np.transpose(np.array([[-math.inf, math.inf] for _ in range(dimensions)]))

        # Build the tree
        return _make_tree_internal(rects, bounds, widths, TreeState(dimensions))

class SingletonTree(DRangeTree):
    def __init__(self, rects):
        self.rects = rects
    def contains(self, point):
        return np.all((point >= self.rects[0]) & (point <= self.rects[1]))
    def dump(self, space):
        print(space, f"singleton {self.rects}")

class ListTree(DRangeTree):
    def __init__(self, rects):
        # Sort rects by size, largest first
        # self.rects = rects[np.argsort(-np.prod(rects[:, 1] - rects[:, 0]))]
        self.rects = rects
    def contains(self, point):
        return np.any(np.all((point >= self.rects[:, 0]) & (point <= self.rects[:, 1]), axis=1))
    def dump(self, space):
        print(space, f"list ({len(self.rects)}) {self.rects[:5]}...")

class EmptyTree(DRangeTree):
    def contains(self, point):
        return False
    def dump(self, space):
        print(space, "empty")

class FullTree(DRangeTree):
    def contains(self, point):
        return True
    def dump(self, space):
        print(space, "full")

class FulfilledTree(DRangeTree):
    def __init__(self, subtree: DRangeTree, axis: int):
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

class SplitDTree(DRangeTree):
    def __init__(self, left: DRangeTree, right: DRangeTree, axis: int, value: float):
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

"""
Remove the j-th column from the final dimension of items.
"""
def without(items, j):
    axis = items.ndim - 1
    indices = [k for k in range(items.shape[axis]) if k != j]
    return np.take(items, indices, axis=axis)

def _make_tree_internal(rects, bounds, widths, state: TreeState):
    """
    Internal function to make a DRangeTree for the hyper-rectangles specified.

    Args:
        rects: hyper-rectangles as a numpy array with dimensions N*2*d, where d is the number of dimensions.
        bounds: the current edges of the search area as a hyper-rectangle as a numpy array of 2*d.
        widths: the widths of each rectangle in each dimension as a numpy array of d.
        state: an internal object for tracking state, used for debugging and developing.
    """

    if len(rects) == 0:
        return EmptyTree()
    if len(rects) == 1:
        return SingletonTree(rects[0])
    if len(rects) < 30:
        return ListTree(rects)

    dimensions = rects.shape[2]

    # If all rects completely fill bounds in a dimension, then that dimension is fulfilled (no further tests needed on it).
    for d in range(dimensions):
        if np.all((rects[:, 0, d] <= bounds[0, d]) & (rects[:, 1, d] >= bounds[1, d])):
            if rects.shape[2] == 1:
                return FullTree()
            else:
                sub_rects = np.unique(without(rects, d), axis=0)
                sub_bounds = without(bounds, d)
                sub_widths = without(widths, d)
                subtree = _make_tree_internal(sub_rects, sub_bounds, sub_widths, state.drop(d))
                return FulfilledTree(subtree, d)

    # Identify possible split points for each dimension. Logically, these are the edges of the hyper-rects,
    # as it doesn't make sense to split not on an edge.
    splits = [np.unique(values) for values in (rects[:, :, j] for j in range(dimensions))]
    
    # Find the dimension with the most variety with respect to its width.
    best_d = 0
    best_classes = 0
    for d in range(dimensions):
        min = np.min(splits[d])
        max = np.max(splits[d])
        r = max - min
        max_classes = r / widths[d]
        # TODO: Hoist len(split) < 3 check to here
        if max_classes > best_classes:
            best_d = d
            best_classes = max_classes

    if best_classes < 0.9: # Diminishing returns as this parameter is dropped; this seems like reasonable trade-off
        # Wherever we split we're hardly going to achieve much, so fall out to a list
        return ListTree(rects)

    d = best_d
    split = splits[d]

    # We want to split this dimension into two spaces. Given the outermost points don't
    # actually reduce the number of nodes (because everything is on one side of the outermost point)
    # if there are less than three split points (i.e. no inner split points) there is no point
    # splitting, so fall back to a list.
    if len(split) < 3:
        print(f"WARNING: Can't split as {split} has <3 members, falling back to list")
        return ListTree(rects)
    
    # This is slow but worth it to find the "best" split.
    # We score a split as the maximum of items on either side of the split, divided by the total items.
    # The intuition is that a 80/20 split is bad, as is a 75/75 split. The best split is the closest
    # to 50/50. score can never drop below half the length of list, as then we'd be losing items somewhere.
    # (Intuitively, scale score by len(rects))
    # TODO: max(split) + excess(min(split)) might be a better score. So 80/20 should score 0.8, which would be
    #       better than 75/75 which currently scores 0.75 but would score 0.75+0.25=1.0. Max score will become
    #       1.5 for a 100/100 split, so we could subtract 0.5 to get this to scale from 0-1.
    #       Simpler calculation would then be excess(left) + excess(right), which feels plausible. 

    # Intially, we'll assume the best split is in the middle.
    # We give this a score as if it was a 60/60 split; that might not be realistic and might need revising,
    # but the intuition is that if we can't do better than a 60/60 split, we might as well just cut down
    # the middle which might free up other cuts better than some lop-sided cut.
    best_split_pos = len(split) // 2
    best_score = len(rects) * 0.6

    lefts = rects[:, 0, d]
    rights = rects[:, 1, d]
    for split_pos in range(len(split)):
        split_at = split[split_pos]
        left_count = (lefts < split_at).sum()
        right_count = (rights > split_at).sum()
        score = left_count if left_count > right_count else right_count
        if best_score is None or score < best_score:
            best_score = score
            best_split_pos = split_pos

    # Record the point we split at
    split_at = split[best_split_pos]

    # Split and clip the rectangles at the split point
    # Rectangles do not include their end points in either direction.
    # We clip to make split point calculations accurate later on.
    lefts = rects[lefts < split_at]
    lefts[:, 1, d] = np.clip(lefts[:, 1, d], a_max = split_at, a_min = None)
    lefts = np.unique(lefts, axis = 0)
    lefts = lefts[lefts[:, 0, d] < lefts[:, 1, d]] # Filter out empty rectangles

    rights = rects[rights > split_at]
    rights[:, 0, d] = np.clip(rights[:, 0, d], a_min = split_at, a_max = None)
    rights = np.unique(rights, axis = 0)
    rights = rights[rights[:, 0, d] < rights[:, 1, d]] # Filter out empty rectangles

    # Update the bounds we're tracking to know about the split we've just made.
    left_bounds = np.copy(bounds)
    left_bounds[1, d] = split_at

    right_bounds = np.copy(bounds)
    right_bounds[0, d] = split_at

    # Build the subtrees
    left = _make_tree_internal(lefts, left_bounds, widths, state.descend(d))
    right = _make_tree_internal(rights, right_bounds, widths, state.descend(d))
    return SplitDTree(left, right, d, split_at)

def _self_test():
    import pandas as pd
    from methods.common.luc import luc_matching_columns
    from collections import defaultdict
    from time import time
    import timeit

    luc0, luc5, luc10 = luc_matching_columns(2012)

    source_pixels = pd.read_parquet("./test/data/1201-k.parquet")
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
    np.random.seed(42)
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
    
    print("making tree... (this will take a few seconds)")
    start = time()
    tree = DRangeTree.build(sources, np.array([elevation_width, slope_width, access_width]))
    print("build time", time() - start)
    print("tree depth", tree.depth())
    print("tree size", tree.size())

    def do_nd_tree_matching():
        found = 0
        for i in range(length):
            found += 1 if tree.contains(np.array([elevation[i], slopes[i], access[i]])) else 0
        return found
    
    def speed_of(what, func):
        assert(func() == 314) # If you change the random seed, change this.
        t = timeit.Timer(stmt=func)
        loops, time = t.autorange()
        print(what, ": ", time / loops, "per call")
    speed_of("NP matching", do_np_matching)
    speed_of("Tree matching", do_nd_tree_matching)

if __name__ == "__main__":
    _self_test()
