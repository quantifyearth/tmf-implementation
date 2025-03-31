import math

import numpy as np
from numba import float32, int32 # type: ignore
from numba.experimental import jitclass # type: ignore


class KDTree:
    """
    A k-d tree represents points in a K-dimensional space.
    
    We expect to do range searches to find points that match a range on all k dimensions.
    """
    def __init__(self):
        pass

    def contains(self, range) -> bool:
        """Does the tree contain a point in range?"""
        raise NotImplementedError()

    def depth(self) -> int:
        """The height of the deepest node in the tree."""
        return 1

    def size(self) -> int:
        """The number of nodes in the tree."""
        return 1

    def count(self) -> int:
        """The number of points in the tree."""
        return 0

    def members(self, range) -> np.ndarray:
        """Return a list of all members in range."""
        raise NotImplementedError()

    def dump(self, _space: str):
        """Return a string representation of the tree for debugging."""
        raise NotImplementedError()

class KDLeaf(KDTree):
    """
    A leaf repesents a single point in the tree.
    """
    def __init__(self, point, index):
        self.point = point
        self.index = index
    def contains(self, range) -> bool:
        return np.all(range[0] <= self.point) & np.all(range[1] >= self.point) # type: ignore
    def members(self, range):
        if self.contains(range):
            return np.array([self.index])
        return np.empty(0, dtype=np.int_)
    def dump(self, space: str):
        print(space, f"point {self.point}")
    def count(self):
        return 1

class KDList(KDTree):
    """
    A list node repesents a list of points in the tree.

    This is an optimisation for when linear search becomes quicker than walking the tree.
    """
    def __init__(self, points, indexes):
        self.points = points
        self.indexes = indexes
    def contains(self, range) -> bool:
        return np.any(np.all(range[0] <= self.points, axis=1) & np.all(range[1] >= self.points, axis=1)) # type: ignore
    def members(self, range):
        return self.indexes[np.all(range[0] <= self.points, axis=1) & np.all(range[1] >= self.points, axis=1)]
    def dump(self, space: str):
        print(space, f"points {self.points}")
    def count(self):
        return len(self.points)

class KDSplit(KDTree):
    """
    A split node represents am axis-aligned binary split in the tree.

    The tree splits into tree disjoint subtrees at value in dimension d, called left and right.
    All the functions that take a range must consider the need to look into both left and right
    if the range intersects the split point.
    """
    def __init__(self, d: int, value: float, left: KDTree, right: KDTree):
        self.d = d
        self.value = value
        self.left = left
        self.right = right
    def contains(self, range) -> bool:
        l = self.value - range[0, self.d] # Amount on left side
        r = range[1, self.d] - self.value # Amount on right side
        # Either l or r must be positive, or both
        # Pick the biggest first
        if l >= r:
            if self.left.contains(range):
                return True
            # Visit the rest if it is inside
            if r >= 0:
                if self.right.contains(range):
                    return True
        else:
            if self.right.contains(range):
                return True
            # Visit the rest if it is inside
            if l >= 0:
                if self.left.contains(range):
                    return True
        return False

    def members(self, range):
        result = None
        if self.value >= range[0, self.d]:
            result = self.left.members(range)
        if range[1, self.d] >= self.value:
            rights = self.right.members(range)
            if result is None:
                result = rights
            else:
                result = np.append(result, rights, axis=0)
        return result if result is not None else np.empty(0, dtype=np.int_)

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def dump(self, space: str):
        print(space, f"split d{self.d} at {self.value}")
        print(space + "  <")
        self.left.dump(space + "\t")
        print(space + "  >")
        self.right.dump(space + "\t")
    def count(self):
        return self.left.count() + self.right.count()


class KDRangeTree:
    """Wrap up a KDTree with a fixed width range for queries. 
    """
    def __init__(self, tree, widths):
        self.tree = tree
        self.widths = widths
    def contains(self, point) -> bool:
        return self.tree.contains(np.array([point - self.widths, point + self.widths]))
    def members(self, point) -> np.ndarray:
        return self.tree.members(np.array([point - self.widths, point + self.widths]))
    def dump(self, space: str):
        self.tree.dump(space)
    def size(self):
        return self.tree.size()
    def depth(self):
        return self.tree.depth()
    def count(self):
        return self.tree.count()

def make_kdrangetree(points, widths):
    """Make a KDRangeTree containing points that is queried with ranges of width width.
    
    We recursively split up the data by the most-discriminating dimension.
    If no dimension splits the data up very much, or there are less than a cut-off number of
    points, we output a list node instead of a split.
    """
    def make_kdtree_internal(points, indexes):
        if len(points) == 1:
            return KDLeaf(points[0], indexes[0])
        if len(points) < 30:
            return KDList(points, indexes)
        # Find split in dimension with most bins
        dimensions = points.shape[1]
        bins: float = None # type: ignore
        chosen_d_min = 0
        chosen_d_max = 0
        chosen_d = 0
        for d in range(dimensions):
            d_max = np.max(points[:, d])
            d_min = np.min(points[:, d])
            d_range = d_max - d_min
            d_bins = d_range / widths[d]
            if bins is None or d_bins > bins:
                bins = d_bins
                chosen_d = d
                chosen_d_max = d_max
                chosen_d_min = d_min

        if bins < 1.3:
            # No split is very worthwhile, so dump points
            return KDList(points, indexes)

        split_at = np.median(points[:, chosen_d])
        # Avoid degenerate cases
        if split_at == chosen_d_max or split_at == chosen_d_min:
            split_at = (chosen_d_max + chosen_d_min) / 2

        left_side = points[:, chosen_d] <= split_at
        right_side = ~left_side
        lefts = points[left_side]
        rights = points[right_side]
        lefts_indexes = indexes[left_side]
        rights_indexes = indexes[right_side]
        return KDSplit(chosen_d, split_at, make_kdtree_internal(lefts, lefts_indexes), make_kdtree_internal(rights, rights_indexes))
    indexes = np.arange(len(points))
    return KDRangeTree(make_kdtree_internal(points, indexes), widths)

@jitclass([('ds', int32[:]),  ('values', float32[:]), ('items', int32[:]), ('lefts', int32[:]), ('rights', int32[:]), ('rows', float32[:, :]), ('dimensions', int32), ('widths', float32[:])])
class RumbaTree:
    """A RumbaTree is a KDRangeTree which is optimised with Numba.
    
    Instead of pointers, the various members of the tree are serialised into arrays to make
    traversal in Numba easy.
    """
    def __init__(self, ds: np.ndarray, values: np.ndarray, items: np.ndarray, lefts: np.ndarray, rights: np.ndarray, rows: np.ndarray, dimensions: int, widths: np.ndarray):
        self.ds = ds
        self.values = values
        self.items = items
        self.lefts = lefts
        self.rights = rights
        self.rows = rows
        self.dimensions = dimensions
        self.widths = widths
    def members(self, point: np.ndarray):
        """
        Return all the items that are within widths of point.

        We use a queue to start at the top of the tree and, for each node, decide if the left or
        right need to be processed, adding them to the queue if so.
        For leaf nodes (marked with a value of NaN), we instead process the list of items at that
        node and return all of the items that are within widths of point.
        """
        low = point - self.widths
        high = point + self.widths
        queue = [0]
        finds = []
        while len(queue) > 0:
            pos = queue.pop()
            d = self.ds[pos]
            value = self.values[pos]
            if math.isnan(value):
                i = d
                item = self.items[i]
                while item != -1:
                    # Check item
                    found = True
                    for d in range(self.dimensions):
                        value = self.rows[item, d]
                        if value < low[d]:
                            found = False
                            break
                        if value > high[d]:
                            found = False
                            break
                    if found:
                        finds.append(item)
                    i += 1
                    item = self.items[i]
            else:
                if value <= high[d]:
                    queue.append(self.rights[pos])
                if value >= low[d]:
                    queue.append(self.lefts[pos])
        return finds
    def count_members(self, point: np.ndarray):
        """
        Return a count of items that are within widths of point.

        Mostly just for debugging.
        """
        low = point - self.widths
        high = point + self.widths
        queue = [0]
        count = 0
        while len(queue) > 0:
            pos = queue.pop()
            d = self.ds[pos]
            value = self.values[pos]
            if math.isnan(value):
                i = d
                item = self.items[i]
                while item != -1:
                    # Check item
                    found = True
                    for d in range(self.dimensions):
                        value = self.rows[item, d]
                        if value < low[d]:
                            found = False
                            break
                        if value > high[d]:
                            found = False
                            break
                    if found:
                        count += 1
                    i += 1
                    item = self.items[i]
            else:
                if value <= high[d]:
                    queue.append(self.rights[pos])
                if value >= low[d]:
                    queue.append(self.lefts[pos])
        return count
    def members_sample(self, point: np.ndarray, count: int, rng: np.random.Generator):
        """
        Return up to count items that are within widths of point, selected at random.

        As with members, we use a queue to start at the top of the tree and, for each node, decide
        if the left or right need to be processed, adding them to the queue if so.

        However, for leaf nodes, instead of taking all items that match, we use algorithm R for
        resoivoir sampling to select up to count items at random.
        https://en.wikipedia.org/wiki/Reservoir_sampling#Simple:_Algorithm_R
        """
        low = point - self.widths
        high = point + self.widths
        queue = [0]
        finds: list[int] = []
        # We need to track how many items have been found for the sampling.
        found_count = 0
        # rng is noticable slow, so we seed 256-bit inline xoshiro generator here.
        rand_state = rng.integers(0, 0xFFFF_FFFF_FFFF_FFFF, 4, np.uint64, True)
        while len(queue) > 0:
            pos = queue.pop()
            d = self.ds[pos]
            value = self.values[pos]
            if math.isnan(value):
                i = d
                item: int = self.items[i] # type: ignore
                while item != -1:
                    # Check item
                    found = True
                    for d in range(self.dimensions):
                        value = self.rows[item, d]
                        if value < low[d]:
                            found = False
                            break
                        if value > high[d]:
                            found = False
                            break
                    if found:
                        # Keep track of how many items are found for the probabilities to work out.
                        found_count += 1
                        # For each item we find, decide here whether to include it in the output.
                        if len(finds) < count:
                            # If we haven't found count items yet, we provisionaly take this item.
                            finds.append(item)
                        else:
                            # Replace a random item in finds based on algorithm R.

                            # When we find the N+1th item, by induction, we assume we have selected
                            # count of the previous N items with 1/N probability.
                            #
                            # We want to select this item with count/N+1 probability
                            # and we want it to replace one of the previously selected
                            # items with a uniform 1/count probability.
                            # We generate a uniform random number, pos, from [0, N+1)
                            # and select this item if that number is less than count
                            # (which corresponds to our count/N+1 probability of selection).
                            # The position to replace the item is then given by pos,
                            # given we now know it is a uniform random number in [0, count)
                            # (which again matches our desired distribution).

                            # Generate a random number
                            # Source: https://prng.di.unimi.it/xoshiro256plus.c
                            rand = rand_state[0] + rand_state[3]
                            t = rand_state[1] << 17
                            rand_state[2] ^= rand_state[0]
                            rand_state[3] ^= rand_state[1]
                            rand_state[1] ^= rand_state[2]
                            rand_state[0] ^= rand_state[3]
                            rand_state[2] ^= t
                            rand_state[3] = (rand_state[3] >> 45) | (rand_state[3] << 19)
                            
                            # Work out where this item should fall.
                            pos = rand % found_count

                            # Replace an existing item if we should include this item.
                            if pos < count:
                                finds[pos] = item
                    i += 1
                    item = self.items[i] # type: ignore
            else:
                if value <= high[d]:
                    queue.append(self.rights[pos])
                if value >= low[d]:
                    queue.append(self.lefts[pos])
        return finds

NAN = float('nan')
def make_rumba_tree(tree: KDRangeTree, rows: np.ndarray):
    """
    Make a RumbaTree from a KDRangeTree.

    This just involved walking the tree and flattening it out into arrays.
    The code is slightly fiddly because we don't know how big a node is until we have walked it,
    so we output a placeholder for the right node and update it after walking the left node.
    """
    ds: list[int] = []
    values = []
    items: list[int] = []
    lefts = []
    rights = []
    widths = None
    def recurse(node):
        nonlocal widths
        if isinstance(node, KDSplit):
            pos = len(ds)
            ds.append(node.d)
            values.append(node.value)
            lefts.append(pos + 1) # Next node we output will be left
            rights.append(0xDEADBEEF) # Put placeholder here...
            recurse(node.left)
            rights[pos] = len(ds) # ..and fixup right to be the next node we output
            recurse(node.right)
        elif isinstance(node, KDList):
            values.append(NAN)
            ds.append(len(items))
            lefts.append(-1) # Specific invalid values for debugging an errors in tree build
            rights.append(-2)
            for item in node.indexes:
                items.append(item)
            items.append(-1)
        elif isinstance(node, KDLeaf):
            values.append(NAN)
            ds.append(len(items))
            lefts.append(-3)
            rights.append(-4)
            items.append(node.index)
            items.append(-1)
        elif isinstance(node, KDRangeTree):
            widths = node.widths
            recurse(node.tree)
    recurse(tree)
    if widths is None:
        raise ValueError(f"Expected KDRangeTree, got {tree}")
    return RumbaTree(
        np.array(ds, dtype=np.int32),
        np.array(values, dtype=np.float32),
        np.array(items, dtype=np.int32),
        np.array(lefts, dtype=np.int32),
        np.array(rights, dtype=np.int32),
        np.ascontiguousarray(rows, dtype=np.float32),
        rows.shape[1],
        np.ascontiguousarray(widths, dtype=np.float32),
    )
