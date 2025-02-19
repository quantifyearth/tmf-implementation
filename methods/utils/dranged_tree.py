"""
This file and associated classes represent a dimensional-ranged-tree, which is a name I made up.
The idea of a d-ranged tree is you can build it from a list of N-dimensional ranges,
and then query points for membership. It is like a k-d tree, but items can end up
clipped on both sides of a partition. This is a time-memory trade-off in favour of
quicker search times.

Ranges can be specified in absolute or relative terms. This also makes this structure easier to query than a k-d tree,
whilst also being 10x faster to query - the reason being, our ranges overlap a lot, so a structure that merges
those ranges together will be faster than a k-d tree which knows nothing about the overlapped structure.
"""

from __future__ import annotations

import numpy as np
import math

class DRangedTree:
    def contains(self, point: np.ndarray):
        raise NotImplementedError

    def depth(self) -> int:
        return 1

    def dump(self, space: str) -> None:
        raise NotImplementedError

    def size(self) -> int:
        return 1

    @staticmethod
    def build(items: np.ndarray, widths: np.ndarray, expected_fraction: float) -> DRangedTree:
        """
        Build a DRangedTree for the items in list. Each item has the corresponding width for each dimension +/- to it.
        If the width for a dimension is negative, it is treated as a relative fractional width.
        So a width of 10 means the value of item +/- 10, whereas a width of -0.1 means +/- 10% of the original value.
        expected_fraction is the proportion of search points we expect to match with.
        """
        items = np.unique(items, axis = 0) # Ditch duplicate points
        # Reshape each point into a hyper-rect +- width
        rects = []
        for item in items:
            lefts = []
            rights = []
            for d, value in enumerate(item):
                width = widths[d]
                if width < 0:
                    fraction = -width
                    width = value * fraction
                lefts.append(value - width)
                rights.append(value + width)
            rects.append([lefts, rights])
        # Calculate initial bounds
        dimensions = items.shape[items.ndim - 1]
        bounds = np.transpose(np.array([[-math.inf, math.inf] for _ in range(dimensions)]))

        # Build the tree
        state = TreeState(dimensions)
        state.logging = False
        state.bound_dimension_at = math.ceil(math.log2(1/(1-math.pow(expected_fraction, 1 / len(widths))))) - 1
        return _make_tree_internal(np.array(rects), bounds, state)

class SingletonTree(DRangedTree):
    def __init__(self, rects):
        self.rects = rects
    def contains(self, point):
        return np.all((point >= self.rects[0]) & (point <= self.rects[1]))
    def dump(self, space):
        print(space, f"singleton {self.rects}")

class ListTree(DRangedTree):
    def __init__(self, rects):
        self.rects = rects
    def contains(self, point):
        return np.any(np.all((point >= self.rects[:, 0]) & (point <= self.rects[:, 1]), axis=1))
    def dump(self, space):
        print(space, f"list ({len(self.rects)}) {self.rects[:5]}...")

class EmptyTree(DRangedTree):
    def contains(self, point):
        return False
    def dump(self, space):
        print(space, "empty")

class FullTree(DRangedTree):
    def contains(self, point):
        return True
    def dump(self, space):
        print(space, "full")

class FulfilledTree(DRangedTree):
    def __init__(self, subtree: DRangedTree, axis: int):
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

class CheckTree(DRangedTree):
    def __init__(self, axis: int, value: float, subtree: DRangedTree, continuation: DRangedTree|None = None):
        self.axis = axis
        self.value = value
        self.subtree = subtree
        self.continuation = continuation
    def contains(self, point):
        if point[self.axis] == self.value:
            return self.subtree.contains(without(point, self.axis))
        elif self.continuation:
            return self.continuation.contains(point)
    def dump(self, space):
        print(space, f"check axis {self.axis} == {self.value} ->")
        self.subtree.dump(space + "\t")
        if self.continuation:
            print(space, f"  else")
            self.continuation.dump(space + "\t")
    def depth(self):
        return 1 + max(self.subtree.depth(), self.continuation.depth() if self.continuation else 0)
    def size(self):
        return 1 + self.subtree.size() + (self.continuation.size() if self.continuation else 0)

class SplitDTree(DRangedTree):
    def __init__(self, left: DRangedTree, right: DRangedTree, axis: int, value: float):
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
        print(space + "  <=")
        self.left.dump(space + "\t")
        print(space + "  >")
        self.right.dump(space + "\t")
    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())
    def size(self):
        return 1 + self.left.size() + self.right.size()

class SplitDLeanRightTree(DRangedTree):
    """This is identical to a SplitDTree, except values equal to the value go into the right tree instead of the left"""
    def __init__(self, left: DRangedTree, right: DRangedTree, axis: int, value: float):
        self.left = left
        self.right = right
        self.axis = axis
        self.value = value
    def contains(self, point):
        if point[self.axis] >= self.value:
            return self.right.contains(point)
        else:
            return self.left.contains(point)
    def dump(self, space):
        print(space, f"split axis {self.axis} at {self.value}")
        print(space + "  <")
        self.left.dump(space + "\t")
        print(space + "  >=")
        self.right.dump(space + "\t")
    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())
    def size(self):
        return 1 + self.left.size() + self.right.size()

class TreeState:
    depth: int
    dimensions: int
    descent: np.ndarray
    logging: bool
    bound_dimension_at: int
    def __init__(self, dimensions_or_tree: int|TreeState, j: int = -1, drop: bool = False):
        if isinstance(dimensions_or_tree, TreeState):
            existing = dimensions_or_tree
            assert(j < existing.dimensions)
            self.depth = existing.depth + 1
            self.logging = existing.logging
            self.bound_dimension_at = existing.bound_dimension_at
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
            self.logging = False
            self.bound_dimension_at = -1

    def descend(self, j: int) -> TreeState:
        return TreeState(self, j)

    def drop(self, j: int) -> TreeState:
        return TreeState(self, j, drop=True)

    def print(self, s: str) -> None:
        if self.logging:
            print(self.depth * "  ", s)

"""
Remove the j-th column from the final dimension of items.
"""
def without(items, j):
    axis = items.ndim - 1
    indices = [k for k in range(items.shape[axis]) if k != j]
    return np.take(items, indices, axis=axis)

def _make_tree_internal(rects: np.ndarray, bounds: np.ndarray, state: TreeState):
    """
    Internal function to make a DRangeTree for the hyper-rectangles specified.

    Args:
        rects: hyper-rectangles as a numpy array with dimensions N×2×d, where d is the number of dimensions.
        bounds: the current edges of the search area as a hyper-rectangle as a numpy array of 2×d.
        state: an internal object for tracking state of tree build, used for optimisation, debugging and developing.
    """
    state.print(f"T {len(rects)} ∆{state.dimensions}")

    if len(rects) == 0:
        return EmptyTree()
    if len(rects) == 1:
        return SingletonTree(rects[0])
    if len(rects) < 50:
        return ListTree(rects)
    if state.depth == 30:
        print(f"Limiting depth to {state.depth} with {len(rects)} rects remaining")
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
                subtree = _make_tree_internal(sub_rects, sub_bounds, state.drop(d))
                return FulfilledTree(subtree, d)

    # Check if we need to bound a dimension
    for d in range(dimensions):
        if state.descent[d] == state.bound_dimension_at:
            # Check if bounds are infinite
            if bounds[0, d] == -math.inf:
                left = np.min(rects[:, 0, d])
                new_bounds = np.copy(bounds)
                new_bounds[0, d] = left
                subtree = _make_tree_internal(rects, new_bounds, state.descend(d))
                # We need to lean right, because we only want to exclude values strictly less than left.
                return SplitDLeanRightTree(EmptyTree(), subtree, d, left)
            if bounds[1, d] == math.inf:
                right = np.max(rects[:, 1, d])
                new_bounds = np.copy(bounds)
                new_bounds[1, d] = right
                subtree = _make_tree_internal(rects, new_bounds, state.descend(d))
                return SplitDTree(subtree, EmptyTree(), d, right)


    # Identify possible split points for each dimension. Logically, these are the edges of the hyper-rects,
    # as it doesn't make sense to split not on an edge.
    lefts = [rects[:, 0, d] for d in range(dimensions)]
    rights = [rects[:, 1, d] for d in range(dimensions)]
    splits = [np.unique(np.array([lefts[d], rights[d]])) for d in range(dimensions)]
    widths = [rights[d] - lefts[d] for d in range(dimensions)]
    widths_without_zeros = [widths[widths > 0] for widths in widths]

    def make_tree_without_d(d: int, rects):
        if dimensions == 1:
            return FullTree()
        else:
            sub_rects = np.unique(without(rects, d), axis=0)
            sub_bounds = without(bounds, d)
            return _make_tree_internal(sub_rects, sub_bounds, state.drop(d))

    for d in range(dimensions):
        # If a dimension has only one split point, output a single value node check for that dimension
        if len(splits[d]) == 1:
            return CheckTree(d, splits[d][0], make_tree_without_d(d, rects))
        # If a good fraction of this dimension has a single value, drop that value
        needed_values = len(rects) * 0.1
        if (len(widths[d]) - len(widths_without_zeros[d])) > needed_values:
            # Find the single values
            single_valued_points = lefts[d][lefts[d] == rights[d]]
            svp_values, svp_counts = np.unique(single_valued_points, return_counts=True)
            best_index = np.argmax(svp_counts)
            # Swallowing 10% of a dimension is good news
            if svp_counts[best_index] > needed_values:
                value = svp_values[best_index]
                swallowed = (rects[:, 0, d] == value) & (rects[:, 1, d] == value)
                remainder = rects[~swallowed]
                #state.print(f"  found value: {svp_values[best_index]} with {svp_counts[best_index]} instances and remainder of {len(remainder)} out of {len(rects)}")
                if len(remainder) + svp_counts[best_index] != len(rects):
                    raise RuntimeError("Incorrect number items in remainder")
                continuation = _make_tree_internal(remainder, bounds, state.descend(d))
                rects_with_value = rects[swallowed]
                return CheckTree(d, value, make_tree_without_d(d, rects_with_value), continuation)

    # Find the dimension with the most variety with respect to its width.
    best_d = 0
    best_classes = 0
    best_width_estimate = 0
    for d in range(dimensions):
        split = splits[d]
        min_split = np.min(split)
        max_split = np.max(split)
        if len(widths_without_zeros[d]) == 0:
            # If all widths are zero, we can calculate estimate the number of classes as
            # the number of unique split(value) points.
            max_classes = len(split)
            width_estimate = (max_split - min_split) / max_classes
        elif len(split) >= 3:
            # We want to split this dimension into two spaces. Given the outermost points don't
            # actually reduce the number of nodes (because everything is on one side of the outermost point)
            # if there aren't at least three split points (i.e. no inner split points) there is no point
            # splitting.
            width_estimate = np.median(widths_without_zeros[d])
            r = max_split - min_split
            max_classes = r / width_estimate
            best_split_pos = len(split) // 2
            split_at = split[best_split_pos]

            left_count = np.sum(lefts[d] <= split_at)
            right_count = np.sum(rights[d] >= split_at)
            target = len(rects) * 1
            total_target = len(rects) * 1.99
            # FIXME: limitied this to less than 2 breaks splitting which consumes dimensions
            # by putting a split in the middle of a overlap area, which can then be consumed
            # in the following layer on either side.
            # We need a heuristic between layers to consume the overlap, or to detect and generate
            # a 3-tree where there is an overlap.
            if left_count > target or right_count > target or left_count + right_count > total_target:
                max_classes = 0 # Don't split here as the two side trees are too large or unbalanced
        else:
            max_classes = 0 # Not a good dimension to split on
            width_estimate = 0

        if max_classes > best_classes:
            best_d = d
            best_classes = max_classes
            best_width_estimate = width_estimate
    if best_classes < 1.3:
        # Diminishing returns as this parameter is dropped; this seems like reasonable trade-off
        # Wherever we split we're hardly going to achieve much, so fall out to a list
        return ListTree(rects)

    d = best_d
    split = splits[d]

    # We want to split this dimension into two spaces. Given the outermost points don't
    # actually reduce the number of nodes (because everything is on one side of the outermost point)
    # if there are less than three split points (i.e. no inner split points) there is no point
    # splitting, so fall back to a list.
    if len(split) < 3:
        # We can only get here if we're splitting on classes, and there are only two classes,
        # so we can simply split at a point between the two classes.
        split_at = np.mean(split)
    else:
        if False and best_classes < 3:
            # Chance there is an overlap here, so let's check and eliminate it if possible
            # L  L L LL LL
            #                R  R RR R     R
            #            L   R
            #              ^- overlap here which we can consume the dimension in

            left_cut = np.max(lefts[d])
            right_cut = np.min(rights[d])
            overlap_width = right_cut - left_cut
            # Only worth doing if at least 20% overlap.
            # FIXME: currently this code leads to fewer matches in test for unknown reasons
            if overlap_width > 0.2 * best_width_estimate:
                left_rects = np.copy(rects)
                left_rects[:, 1, d] = np.clip(left_rects[:, 1, d], a_max = left_cut, a_min = None)
                left_rects = np.unique(left_rects, axis = 0)

                right_rects = np.copy(rects)
                right_rects[:, 0, d] = np.clip(right_rects[:, 0, d], a_min = right_cut, a_max = None)
                right_rects = np.unique(right_rects, axis = 0)

                middle = without(rects, d)

                state.print(f" swallowing ∂{d} from {left_cut} to {right_cut} from estimated_width {best_width_estimate} of size: {len(rects)}")

                # Update the bounds we're tracking to know about the split we've just made.
                left_bounds = np.copy(bounds)
                left_bounds[1, d] = left_cut

                right_bounds = np.copy(bounds)
                right_bounds[0, d] = right_cut

                # Build the subtrees
                left = _make_tree_internal(left_rects, left_bounds, state.descend(d))
                middle = make_tree_without_d(d, middle)
                right = _make_tree_internal(right_rects, right_bounds, state.descend(d))
                return SplitDTree(left, SplitDTree(middle, right, d, right_cut), d, left_cut)

        # We used to assume we could find a "best" split point that balanced the split.
        # But just going for the median seems to work well and is fast, so that's what we do.
        best_split_pos = len(split) // 2

        # Record the point we split at
        split_at = split[best_split_pos]

    # Split and clip the rectangles at the split point
    # Rectangles include their end points to the left.
    # We clip to make split point calculations accurate later on.
    left_rects = rects[rects[:, 0, d] <= split_at]
    left_rects[:, 1, d] = np.clip(left_rects[:, 1, d], a_max = split_at, a_min = None)
    left_rects = np.unique(left_rects, axis = 0)
    #lefts = lefts[lefts[:, 0, d] < lefts[:, 1, d]] # Filter out empty rectangles

    right_rects = rects[rects[:, 1, d] > split_at]
    right_rects[:, 0, d] = np.clip(right_rects[:, 0, d], a_min = split_at, a_max = None)
    right_rects = np.unique(right_rects, axis = 0)
    #rights = rights[rights[:, 0, d] < rights[:, 1, d]] # Filter out empty rectangles

    #state.print(f" splitting ∂{d} at {split_at} (of {len(splits[d])} splits: {splits[d][:10]}) lefts: {len(lefts)} rights: {len(rights)}")

    # Update the bounds we're tracking to know about the split we've just made.
    left_bounds = np.copy(bounds)
    left_bounds[1, d] = split_at

    right_bounds = np.copy(bounds)
    right_bounds[0, d] = split_at

    # Build the subtrees
    left = _make_tree_internal(left_rects, left_bounds, state.descend(d))
    right = _make_tree_internal(right_rects, right_bounds, state.descend(d))
    return SplitDTree(left, right, d, split_at) # type: ignore
