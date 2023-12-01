import argparse
import os
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit, float32, int64, deferred_type  # type: ignore
from numba.experimental import jitclass

import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns

REPEAT_MATCH_FINDING = 1
DEFAULT_DISTANCE = 10000000.0
DEBUG = False

DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "cpc0_u", "cpc0_d",
    "cpc5_u", "cpc5_d",
    "cpc10_u", "cpc10_d"
]
HARD_COLUMN_COUNT = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_match_iteration(
    k_parquet_filename: str,
    m_parquet_filename: str,
    start_year: int,
    output_folder: str,
    idx_and_seed: tuple[int, int]
) -> None:
    logging.info("Find match iteration %d of %d", idx_and_seed[0] + 1, REPEAT_MATCH_FINDING)
    rng = np.random.default_rng(idx_and_seed[1])

    logging.info("Loading K from %s", k_parquet_filename)

    k_set = pd.read_parquet(k_parquet_filename)
    # Methodology 6.5.7: For a 10% sample of K
    k_subset = k_set.sample(
        frac=0.1,
        random_state=rng
    ).reset_index()

    logging.info("Loading M from %s", m_parquet_filename)
    m_set = pd.read_parquet(m_parquet_filename)

    # get the column ids for DISTANCE_COLUMNS
    thresholds_for_columns = np.array([
            200.0, # Elev
            2.5, # Slope
            10.0,  # Access
            0.1, # CPCs
            0.1, # CPCs
            0.1, # CPCs
            0.1, # CPCs
            0.1, # CPCs
            0.1, # CPCs
    ])

    logging.info("Preparing s_set...")

    m_dist_thresholded_df = m_set[DISTANCE_COLUMNS] / thresholds_for_columns
    k_set_dist_thresholded_df = k_set[DISTANCE_COLUMNS] / thresholds_for_columns
    # IDEA: Maybe we can bin these somehow?
    
    # Rearrange columns by variance so we throw out the least likely to match first
    # except the bottom three which are deforestation CPCs and have more cross-variance between K and M
    variances = np.std(m_dist_thresholded_df, axis=0)
    cols = DISTANCE_COLUMNS
    order = np.argsort(-variances.to_numpy())
    order = np.roll(order, 3)
    new_cols = [cols[o] for o in order]
    m_dist_thresholded_df = m_dist_thresholded_df[new_cols]
    k_set_dist_thresholded_df = k_set_dist_thresholded_df[new_cols]

    # convert to float32 numpy arrays and make them contiguous for numba to vectorise
    m_dist_thresholded = np.ascontiguousarray(m_dist_thresholded_df, dtype=np.float32)
    k_set_dist_thresholded = np.ascontiguousarray(k_set_dist_thresholded_df, dtype=np.float32)

    # LUC columns are all named with the year in, so calculate the column names
    # for the years we are intested in
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    # As well as all the LUC columns for later use
    luc_columns = [x for x in m_set.columns if x.startswith('luc')]

    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]
    assert len(hard_match_columns) == HARD_COLUMN_COUNT

    # similar to the above, make the hard match columns contiguous float32 numpy arrays
    m_dist_hard = np.ascontiguousarray(m_set[hard_match_columns].to_numpy()).astype(np.int32)
    k_set_dist_hard = np.ascontiguousarray(k_set[hard_match_columns].to_numpy()).astype(np.int32)

    # Methodology 6.5.5: S should be 10 times the size of K
    required = 10

    logging.info("Running make_s_set_mask... required: %d", required)
    starting_positions = rng.integers(0, int(m_dist_thresholded.shape[0]), int(k_set_dist_thresholded.shape[0]))
    s_set_mask_true, no_potentials = make_s_set_mask(
        m_dist_thresholded,
        k_set_dist_thresholded,
        m_dist_hard,
        k_set_dist_hard,
        starting_positions,
        required,
        rng
    )

    logging.info("Done make_s_set_mask. s_set_mask.shape: %a", {s_set_mask_true.shape})

    s_set = m_set[s_set_mask_true]
    potentials = np.invert(no_potentials)

    # FIXME: Not sure this line is meaningful any more if potentials drawn from K?
    k_subset = k_subset[potentials]
    logging.info("Finished preparing s_set. shape: %a", {s_set.shape})

    # Notes:
    # 1. Not all pixels may have matches
    results = []
    matchless = []

    s_set_for_cov = s_set[DISTANCE_COLUMNS]
    logging.info("Calculating covariance...")
    covarience = np.cov(s_set_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience).astype(np.float32)

    # Match columns are luc10, luc5, luc0, "country" and "ecoregion"
    s_set_match = s_set[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    s_set_match = np.ascontiguousarray(s_set_match)

    # Now we do the same thing for k_subset
    k_subset_match = k_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    k_subset_match = np.ascontiguousarray(k_subset_match)

    logging.info("Starting greedy matching... k_subset_match.shape: %s, s_set_match.shape: %s",
                 k_subset_match.shape, s_set_match.shape)

    add_results, k_idx_matchless = greedy_match(
        k_subset_match,
        s_set_match,
        invconv
    )

    logging.info("Finished greedy matching...")

    logging.info("Starting storing matches...")

    for result in add_results:
        (k_idx, s_idx) = result
        k_row = k_subset.iloc[k_idx]
        match = s_set.iloc[s_idx]

        if DEBUG:
            for hard_match_column in hard_match_columns:
                if k_row[hard_match_column] != match[hard_match_column]:
                    print(k_row)
                    print(match)
                    raise ValueError("Hard match inconsistency")

        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + DISTANCE_COLUMNS] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + DISTANCE_COLUMNS]
        )

    logging.info("Finished storing matches...")

    for k_idx in k_idx_matchless:
        k_row = k_subset.iloc[k_idx]
        matchless.append(k_row)

    columns = ['k_lat', 'k_lng'] + \
        [f'k_{x}' for x in luc_columns + DISTANCE_COLUMNS] + \
        ['s_lat', 's_lng'] + \
        [f's_{x}' for x in luc_columns + DISTANCE_COLUMNS]

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}.parquet'))

    matchless_df = pd.DataFrame(matchless, columns=k_set.columns)
    matchless_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}_matchless.parquet'))

    logging.info("Finished find match iteration")

class RTree:
    def __init__():
        pass

    def contains(self, range) -> bool:
        raise NotImplemented()

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1
    
    def members(self, range) -> [np.ndarray]:
        raise NotImplemented()

    def dump(self, space: str):
        raise NotImplemented()

class RLeaf(RTree):
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

class RList(RTree):
    def __init__(self, points, indexes):
        self.points = points
        self.indexes = indexes
    def contains(self, range) -> bool:
        return np.any(np.all(range[0] <= self.points, axis=1) & np.all(range[1] >= self.points, axis=1)) # type: ignore
    def members(self, range):
        return self.indexes[np.all(range[0] <= self.points, axis=1) & np.all(range[1] >= self.points, axis=1)]
    def dump(self, space: str):
        print(space, f"points {self.points}")

class RSplit(RTree):
    def __init__(self, d: int, value: float, left: RTree, right: RTree, width: int):
        self.d = d
        self.value = value
        self.left = left
        self.right = right
        self.width = width
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
        l = self.value - range[0, self.d] # Amount on left side
        r = range[1, self.d] - self.value # Amount on right side
        result = None
        if l >= 0:
            result = self.left.members(range)
        if r >= 0:
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

class RWrapper(RTree):
    def __init__(self, tree, widths):
        self.tree = tree
        self.widths = widths
    def contains(self, point) -> bool:
        return self.tree.contains(np.array([point - self.widths, point + self.widths]))
    def members(self, point) -> bool:
        return self.tree.members(np.array([point - self.widths, point + self.widths]))
    def dump(self, space: str):
        self.tree.dump(space)
    def size(self):
        return self.tree.size()
    def depth(self):
        return self.tree.depth()

def make_rtree(points):
    def make_rtree_internal(points, indexes):
        if len(points) == 1:
            return RLeaf(points[0], indexes[0])
        if len(points) < 30:
            return RList(points, indexes)
        # Find split in dimension with most bins
        dimensions = points.shape[1]
        bins = None
        chosen_d_min = 0
        chosen_d_max = 0
        chosen_d = 0
        for d in range(dimensions):
            d_max = np.max(points[:, d])
            d_min = np.min(points[:, d])
            d_range = d_max - d_min
            d_bins = d_range
            if bins == None or d_bins > bins:
                bins = d_bins
                chosen_d = d
                chosen_d_max = d_max
                chosen_d_min = d_min
        
        if bins < 1.3:
            # No split is very worthwhile, so dump points
            return RList(points, indexes)
        
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
        return RSplit(chosen_d, split_at, make_rtree_internal(lefts, lefts_indexes), make_rtree_internal(rights, rights_indexes), dimensions)
    indexes = np.arange(len(points))
    return RWrapper(make_rtree_internal(points, indexes), np.ones(points.shape[1]))

def make_s_set_mask(
    m_dist_thresholded: np.ndarray,
    k_set_dist_thresholded: np.ndarray,
    m_dist_hard: np.ndarray,
    k_set_dist_hard: np.ndarray,
    starting_positions: np.ndarray,
    required: int,
    rng: np.random.Generator
):
    # Make a k-d tree for m_dist_thresholded
    # Ignore dist_hard for now...
    m_tree = make_rtree(m_dist_thresholded)
    logging.info("Size: %d", m_tree.size())
    logging.info("Depth: %d", m_tree.depth())

    k_size = k_set_dist_thresholded.shape[0]
    m_size = m_dist_thresholded.shape[0]

    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    for k in range(k_set_dist_thresholded.shape[0]):
        k_row =  k_set_dist_thresholded[k]
        possible_s = m_tree.members(k_row)
        if len(possible_s) == 0:
            k_miss[k] = True
        else:
            samples = min(len(possible_s), required)
            chosen_s = rng.choice(possible_s, samples)
            if chosen_s.dtype != np.int_:
                print(possible_s)
                print(chosen_s)
                print(chosen_s.dtype)
            s_include[chosen_s] = True
    
    return s_include, k_miss


@jit(nopython=True, fastmath=True, error_model="numpy")
def make_s_set_mask_old(
    m_dist_thresholded: np.ndarray,
    k_set_dist_thresholded: np.ndarray,
    m_dist_hard: np.ndarray,
    k_set_dist_hard: np.ndarray,
    starting_positions: np.ndarray,
    required: int
):
    k_size = k_set_dist_thresholded.shape[0]
    m_size = m_dist_thresholded.shape[0]

    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    for k in range(k_size):
        matches = 0
        k_row = k_set_dist_thresholded[k, :]
        k_hard = k_set_dist_hard[k]

        for index in range(m_size):
            m_index = (index + starting_positions[k]) % m_size

            m_row = m_dist_thresholded[m_index, :]
            m_hard = m_dist_hard[m_index]

            should_include = True
            
            if should_include:
                for j in range(m_row.shape[0]):
                    if abs(m_row[j] - k_row[j]) > 1.0:
                        should_include = False
                        break
                
                if should_include:
                    for j in range(m_hard.shape[0]):
                        if m_hard[j] != k_hard[j]:
                            should_include = False
                            break

                    if should_include:
                        s_include[m_index] = True
                        matches += 1

                        # Don't find any more M's
                        if matches == required:
                            break

        k_miss[k] = matches == 0

    return s_include, k_miss

# Function which returns a boolean array indicating whether all values in a row are true
@jit(nopython=True, fastmath=True, error_model="numpy")
def rows_all_true(rows: np.ndarray):
    # Don't use np.all because not supported by numba

    # Create an array of booleans for rows in s still available
    all_true = np.ones((rows.shape[0],), dtype=np.bool_)
    for row_idx in range(rows.shape[0]):
        for col_idx in range(rows.shape[1]):
            if not rows[row_idx, col_idx]:
                all_true[row_idx] = False
                break

    return all_true


@jit(nopython=True, fastmath=True, error_model="numpy")
def greedy_match(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray
):
    # Create an array of booleans for rows in s still available
    s_available = np.ones((s_set.shape[0],), dtype=np.bool_)
    total_available = s_set.shape[0]

    results = []
    matchless = []

    s_tmp = np.zeros((s_set.shape[0],), dtype=np.float32)

    for k_idx in range(k_subset.shape[0]):
        k_row = k_subset[k_idx, :]

        hard_matches = rows_all_true(s_set[:, :HARD_COLUMN_COUNT] == k_row[:HARD_COLUMN_COUNT]) & s_available
        hard_matches = hard_matches.reshape(
            -1,
        )

        if total_available > 0:
            # Now calculate the distance between the k_row and all the hard matches we haven't already matched
            s_tmp[hard_matches] = batch_mahalanobis_squared(
                s_set[hard_matches, HARD_COLUMN_COUNT:], k_row[HARD_COLUMN_COUNT:], invcov
            )
            # Find the index of the minimum distance in s_tmp[hard_matches] but map it back to the index in s_set
            if np.any(hard_matches):
                min_dist_idx = np.argmin(s_tmp[hard_matches])
                min_dist_idx = np.arange(s_tmp.shape[0])[hard_matches][min_dist_idx]

                results.append((k_idx, min_dist_idx))
                s_available[min_dist_idx] = False
                total_available -= 1
        else:
            matchless.append(k_idx)

    return (results, matchless)

# optimised batch implementation of mahalanobis distance that returns a distance per row
@jit(nopython=True, fastmath=True, error_model="numpy")
def batch_mahalanobis_squared(rows, vector, invcov):
    # calculate the difference between the vector and each row (broadcasted)
    diff = rows - vector
    # calculate the distance for each row in one batch
    dists = (np.dot(diff, invcov) * diff).sum(axis=1)
    return dists


def find_pairs(
    k_parquet_filename: str,
    m_parquet_filename: str,
    start_year: int,
    seed: int,
    output_folder: str,
    processes_count: int
) -> None:
    logging.info("Starting find pairs")
    os.makedirs(output_folder, exist_ok=True)

    rng = np.random.default_rng(seed)
    iteration_seeds = zip(range(REPEAT_MATCH_FINDING), rng.integers(0, 1000000, REPEAT_MATCH_FINDING))

    with Pool(processes=processes_count) as pool:
        pool.map(
            partial(
                find_match_iteration,
                k_parquet_filename,
                m_parquet_filename,
                start_year,
                output_folder
            ),
            iteration_seeds
        )

def main():
    # If you use the default multiprocess model then you risk deadlocks when logging (which we
    # have hit). Spawn is the default on macOS, but not on Linux.
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Takes K and S and finds 100 sets of matches.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_filename",
        help="Parquet file containing pixels from K as generated by calculate_k.py"
    )
    parser.add_argument(
        "--m",
        type=str,
        required=True,
        dest="m_filename",
        help="Parquet file containing pixels from M as generated by build_m_table.py"
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        dest="start_year",
        help="Year project started."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        dest="seed",
        help="Random number seed, to ensure experiments are repeatable."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_directory_path",
        help="Directory into which output matches will be written. Will be created if it does not exist."
    )
    parser.add_argument(
        "-j",
        type=int,
        required=False,
        default=round(cpu_count() / 2),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    find_pairs(
        args.k_filename,
        args.m_filename,
        args.start_year,
        args.seed,
        args.output_directory_path,
        args.processes_count
    )

if __name__ == "__main__":
    main()
