import argparse
import atexit
import math
from multiprocessing.shared_memory import SharedMemory
import os
import logging
from functools import partial
from multiprocessing import Manager, Pool, cpu_count, set_start_method
from queue import Queue
from typing import Iterable

NUMBA_CAPTURED_ERRORS='new_style'
import numpy as np
import pandas as pd
from numba import jit
from pandas import DataFrame

from methods.common.luc import luc_matching_columns
from methods.utils.kd_tree import RumbaTree, make_kdrangetree, make_rumba_tree

REPEAT_MATCH_FINDING = 100
DEFAULT_DISTANCE = 10000000.0
DEBUG = False

DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "cpc0_u", "cpc0_d",
    "cpc5_u", "cpc5_d",
    "cpc10_u", "cpc10_d"
]
HARD_COLUMN_COUNT = 5
THRESHOLDS_FOR_COLUMNS = np.array([
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
REQUIRED = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def category_to_numpy(category_name: str):
    return np.array([*map(int, category_name.split(","))])

def to_category_name(category: np.ndarray):
    return ",".join(map(lambda x: str(int(x)), category))

def load_and_process_k(
    k_parquet_filename: str,
    start_year: int,
) -> tuple[DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    logging.info("Loading K from %s", k_parquet_filename)

    k_set = pd.read_parquet(k_parquet_filename)

    # LUC columns are all named with the year in, so calculate the column names
    # for the years we are intested in
    luc0, luc5, luc10 = luc_matching_columns(start_year)

    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]
    assert len(hard_match_columns) == HARD_COLUMN_COUNT

    # Find categories in K
    hard_match_categories = {to_category_name(k[hard_match_columns].to_numpy()) for _, k in k_set.iterrows()}
    hard_match_categories = [*hard_match_categories]

    k_set_dist_thresholded_df = k_set[DISTANCE_COLUMNS] / THRESHOLDS_FOR_COLUMNS
    k_set_dist_thresholded = np.ascontiguousarray(k_set_dist_thresholded_df, dtype=np.float32)

    k_set_dist_thresholded_by = {}
    k_selector_by = {}
    for category in hard_match_categories:
        k_selector = np.all(k_set[hard_match_columns] == category_to_numpy(category), axis=1)
        logging.info("Building K set for category %a count: %d", category, np.sum(k_selector))
        k_set_dist_thresholded_by[category] = k_set_dist_thresholded[k_selector]
        k_selector_by[category] = k_selector
    
    return k_set, k_set_dist_thresholded_by, k_selector_by, hard_match_categories

def load_and_process_m(
    m_parquet_filename: str,
    hard_match_categories: list[str],
    start_year: int,
    rng: np.random.Generator,
    k_size: int,
) -> tuple[DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[RumbaTree]], dict[str, np.ndarray], np.ndarray]:
    logging.info("Loading M from %s", m_parquet_filename)
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]

    m_set = pd.read_parquet(m_parquet_filename)
    m_dist_thresholded_df = m_set[DISTANCE_COLUMNS] / THRESHOLDS_FOR_COLUMNS
    m_dist_thresholded = np.ascontiguousarray(m_dist_thresholded_df, dtype=np.float32)

    m_dist_thresholded_by = {}
    m_selector_by = {}
    rumba_trees_by = {}
    m_lookup_by = {}

    for category in hard_match_categories:
        m_selector = np.all(m_set[hard_match_columns] == category_to_numpy(category), axis=1)
        m_dist_thresholded_by[category] = m_dist_thresholded[m_selector]
        m_selector_by[category] = m_selector
        
        # Build sets of trees
        m_size = m_dist_thresholded_by[category].shape[0]

        m_sets = max(1, min(100, math.floor(m_size // 1e6), math.ceil(m_size / (k_size * REQUIRED * 10))))
        m_step = math.ceil(m_size / m_sets)

        logging.info("Building %d M trees for category %s count: %d", m_sets, category, np.sum(m_selector))

        m_lookup = np.arange(m_size, dtype=np.int32)
        rng.shuffle(m_lookup)
        
        def m_indexes(m_set: int):
            return m_lookup[m_set * m_step:(m_set + 1) * m_step]

        m_trees = [make_kdrangetree(m_dist_thresholded[m_indexes(m_set)], np.ones(m_dist_thresholded.shape[1])) for m_set in range(m_sets)]

        rumba_trees = [make_rumba_tree(m_tree, m_dist_thresholded) for m_tree in m_trees]
        rumba_trees_by[category] = rumba_trees
        m_lookup_by[category] = m_lookup
    
    m_set_for_cov = m_set[DISTANCE_COLUMNS]
    logging.info("Calculating covariance...")
    covarience = np.cov(m_set_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience).astype(np.float32)

    return m_set, m_dist_thresholded_by, m_selector_by, rumba_trees_by, m_lookup_by, invconv

@jit(nopython=True)
def merge_filter(a: np.ndarray, b: np.ndarray):
    i = 0
    j = 0
    output = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            output.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return np.array(output)

@jit(nopython=True)
def merge_is_member(a: np.ndarray, b: np.ndarray):
    i = 0
    j = 0
    output = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            output.append(False)
            i += 1
        elif a[i] == b[j]:
            output.append(True)
            i += 1
            j += 1
        else:
            j += 1
    return np.array(output, dtype=np.bool_)

def find_match_iteration_fast(
    k_size: int,
    hard_match_categories: list[str],
    k_dist_thresholded_by: dict[str, np.ndarray],
    k_selector_by: dict[str, np.ndarray],
    m_dist_thresholded_by: dict[str, np.ndarray],
    m_selector_by: dict[str, np.ndarray],
    m_trees: dict[str, list[RumbaTree]],
    m_lookup_by: dict[str, np.ndarray],
    invconv: np.ndarray,
    output_queue: Queue,
    idx_and_seed: tuple[int, int],
) -> None:
    logging.info("Find match iteration %d of %d", idx_and_seed[0] + 1, REPEAT_MATCH_FINDING)
    rng = np.random.default_rng(idx_and_seed[1])

    # Methodology 6.5.7: For a 10% sample of K
    k_subset_indexes = np.arange(k_size)
    rng.shuffle(k_subset_indexes)
    k_subset_indexes = k_subset_indexes[0:math.ceil(0.1 * k_size)]
    k_subset_mask = np.zeros(k_size, dtype=np.bool_)
    k_subset_mask[k_subset_indexes] = True

    results = []
    matchless = []

    # Split K and M into those categories and create masks
    for category in hard_match_categories:
        logging.info("%d: Running category %a", idx_and_seed[0] + 1, category)
        k_selector = k_selector_by[category]
        m_selector = m_selector_by[category]
        
        # Make masks for each of those pairs
        s_set_mask_true, no_potentials = make_s_set_mask(
            m_dist_thresholded_by[category],
            k_dist_thresholded_by[category],
            m_trees[category],
            m_lookup_by[category],
            REQUIRED,
            rng
        )

        logging.info("%d: s_set_mask_true built, set values %d of expected %d", idx_and_seed[0] + 1, np.sum(s_set_mask_true), len(k_dist_thresholded_by[category])*REQUIRED)

        k_selected_indexes = np.flatnonzero(k_selector)
        k_subset_indexes = np.flatnonzero(k_subset_mask)

        matchless.extend(merge_filter(k_selected_indexes[no_potentials], k_subset_indexes))

        k_subset_match_indexes = merge_filter(k_selected_indexes[~no_potentials], k_subset_indexes)

        k_subset_match_flags = ~no_potentials & merge_is_member(k_selected_indexes, k_subset_indexes)

        k_subset_match = k_dist_thresholded_by[category][k_subset_match_flags]

        s_set_match = m_dist_thresholded_by[category][s_set_mask_true]
        s_set_match_indexes = np.flatnonzero(m_selector)[s_set_mask_true]

        add_results, k_idx_matchless = greedy_match_simple(
            k_subset_match,
            s_set_match,
            invconv
        )

        matchless.extend(k_subset_match_indexes[k_idx_matchless])

        results.extend([[k_subset_match_indexes[r[0]], s_set_match_indexes[r[1]]] for r in add_results])
    
    output_queue.put((idx_and_seed, results, matchless))

def match_storage(
    output_folder: str,
    k_set: DataFrame,
    m_set: DataFrame,
    match_queue: Queue,
):
    # As well as all the LUC columns for later use
    luc_columns = [x for x in m_set.columns if x.startswith('luc')]

    logging.info("Waiting for matches...")

    for _ in range(REPEAT_MATCH_FINDING):
        idx_and_seed, result_idxs, matchless_idxs = match_queue.get()

        logging.info("Storing match iteration %d", idx_and_seed[0])
        logging.info("Got results count %d, matchless count %d", len(result_idxs), len(matchless_idxs))

        results = []
        for result in result_idxs:
            (k_idx, s_idx) = result
            k_row = k_set.iloc[k_idx]
            match = m_set.iloc[s_idx]

            results.append(
                [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + DISTANCE_COLUMNS] + \
                [match.lat, match.lng] + [match[x] for x in luc_columns + DISTANCE_COLUMNS]
            )

        matchless = []
        for k_idx in matchless_idxs:
            k_row = k_set.iloc[k_idx]
            matchless.append(k_row)

        columns = ['k_lat', 'k_lng'] + \
            [f'k_{x}' for x in luc_columns + DISTANCE_COLUMNS] + \
            ['s_lat', 's_lng'] + \
            [f's_{x}' for x in luc_columns + DISTANCE_COLUMNS]

        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}.parquet'))

        matchless_df = pd.DataFrame(matchless, columns=k_set.columns)
        matchless_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}_matchless.parquet'))

def make_s_set_mask(
    m_dist_thresholded: np.ndarray,
    k_set_dist_thresholded: np.ndarray,
    rumba_trees: list[RumbaTree],
    m_lookup: np.ndarray,
    required: int,
    rng: np.random.Generator
):
    k_size = k_set_dist_thresholded.shape[0]
    m_size = m_dist_thresholded.shape[0]

    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    m_sets = len(rumba_trees)
    m_step = math.ceil(m_size / m_sets)

    def m_index(m_set: int, pos: int):
        return m_lookup[m_set * m_step + pos]

    for k in range(k_size):
        k_row =  k_set_dist_thresholded[k]
        m_order = np.arange(m_sets)
        rng.shuffle(m_order)
        possible_s = []
        for m_set in m_order:
            next_possible_s = rumba_trees[m_set].members_sample(k_row, required, rng)
            if possible_s is None:
                possible_s = [m_index(m_set, s) for s in next_possible_s]
            else:
                take = min(required - len(possible_s), len(next_possible_s))
                possible_s[len(possible_s):len(possible_s)+take] = [m_index(m_set, s) for s in next_possible_s[0:take]]
            if len(possible_s) == required:
                break
        if len(possible_s) == 0:
            k_miss[k] = True
        else:
            s_include[possible_s] = True

    return s_include, k_miss

@jit(nopython=True, fastmath=True, error_model="numpy")
def greedy_match_simple(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray
):
    # Create an array of booleans for rows in s still available
    s_available = np.ones((s_set.shape[0],), dtype=np.bool_)
    total_available = s_set.shape[0]

    results = []
    matchless = []

    for k_idx in range(k_subset.shape[0]):
        k_row = k_subset[k_idx, :]

        if total_available > 0:
            # Now calculate the distance between the k_row and all the hard matches we haven't already matched
            distances = batch_mahalanobis_squared(
                s_set[s_available], k_row, invcov
            )
            min_dist_idx = np.argmin(distances)
            min_dist_idx = np.flatnonzero(s_available)[min_dist_idx]

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

def shared_memory_key(names: Iterable[str]) -> str:
    return "_".join(names)

MEMORY_TO_CLOSE: list[SharedMemory] = []
def close_shared_memory():
    for sm in MEMORY_TO_CLOSE:
        sm.close()
atexit.register(close_shared_memory)

def unpack_shared_memory(cols: int, *names: str, dtype) -> np.ndarray:
    name = shared_memory_key(names)
    def make_shape(rows: int):
        if cols == 1:
            return rows
        else:
            return (rows, cols)
    try:
        sm = SharedMemory(name)
        MEMORY_TO_CLOSE.append(sm)
        itemsize = np.dtype(dtype).itemsize
        rows_f = sm.size / itemsize / cols
        rows = math.floor(rows_f)
        if rows != math.floor(rows):
            raise ValueError(f"SharedMemory for {name} is of incorrect size, got fractional rows {rows}")
        result = np.ndarray(make_shape(rows), dtype=dtype, buffer=sm.buf)
        return result
    except FileNotFoundError:
        logging.debug("Shared memory %s is empty",name)
        return np.ndarray(make_shape(0), dtype=dtype)

def unpack_shared_rumba_tree(*names: str) -> RumbaTree:
    ds = unpack_shared_memory(1, *names, "ds", dtype=np.int32)
    items = unpack_shared_memory(1, *names, "items", dtype=np.int32)
    lefts = unpack_shared_memory(1, *names, "lefts", dtype=np.int32)
    rights = unpack_shared_memory(1, *names, "rights", dtype=np.int32)
    
    values = unpack_shared_memory(1, *names, "values", dtype=np.float32)
    rows = unpack_shared_memory(len(DISTANCE_COLUMNS), *names, "rows", dtype=np.float32)
    widths = unpack_shared_memory(1, *names, "widths", dtype=np.float32)
    return RumbaTree(
        ds,
        values,
        items,
        lefts,
        rights,
        rows,
        len(DISTANCE_COLUMNS), # You want to change this, you implement it.
        widths,
    )

def unpack_iteration_shared_memory(
    prefix: str,
    k_size: int,
    categories: list[str],
    m_tree_counts: dict[str, int],
    invconv: np.ndarray,
    output_queue: Queue,
    idx_and_seed: tuple[int, int],
) -> None:
    k_dist_thresholded_by = {category: unpack_shared_memory(len(DISTANCE_COLUMNS), prefix, "k_dist_thresholded_by", category, dtype=np.float32) for category in categories}
    k_selector_by = {category: unpack_shared_memory(1, prefix, "k_selector_by", category, dtype=np.bool_) for category in categories}
    m_dist_thresholded_by = {category: unpack_shared_memory(len(DISTANCE_COLUMNS), prefix, "m_dist_thresholded_by", category, dtype=np.float32) for category in categories}
    m_selector_by = {category: unpack_shared_memory(1, prefix, "m_selector_by", category, dtype=np.bool_) for category in categories}
    m_trees = {category: [unpack_shared_rumba_tree(prefix, "m_trees", category, str(i)) for i in range(m_tree_counts[category])] for category in categories}
    m_lookup_by = {category: unpack_shared_memory(1, prefix, "m_lookup_by", category, dtype=np.int32) for category in categories}

    find_match_iteration_fast(
        k_size,
        categories,
        k_dist_thresholded_by,
        k_selector_by,
        m_dist_thresholded_by,
        m_selector_by,
        m_trees,
        m_lookup_by,
        invconv,
        output_queue,
        idx_and_seed,
    )

MEMORY_TO_UNLINK: list[SharedMemory] = []
def unlink_shared_memory():
    for sm in MEMORY_TO_UNLINK:
        sm.unlink()
atexit.register(unlink_shared_memory)


def pack_shared_memory(value: np.ndarray, *names: str):
    name = shared_memory_key(names)
    size = value.nbytes
    if size > 0:
        sm = SharedMemory(name, True, size)
        MEMORY_TO_UNLINK.append(sm)
        dest = np.ndarray(shape=value.shape, dtype=value.dtype, buffer=sm.buf)
        dest[:] = value[:]

def pack_shared_rumba_tree(tree: RumbaTree, *names: str):
    pack_shared_memory(tree.ds, *names, "ds")
    pack_shared_memory(tree.items, *names, "items")
    pack_shared_memory(tree.lefts, *names, "lefts")
    pack_shared_memory(tree.rights, *names, "rights")
    
    pack_shared_memory(tree.values, *names, "values")
    pack_shared_memory(tree.rows, *names, "rows")
    pack_shared_memory(tree.widths, *names, "widths")

def load_data_into_shared_memory(
    prefix: str,
    categories: list[str],
    k_dist_thresholded_by: dict[str, np.ndarray],
    k_selector_by: dict[str, np.ndarray],
    m_dist_thresholded_by: dict[str, np.ndarray],
    m_selector_by: dict[str, np.ndarray],
    m_trees: dict[str, list[RumbaTree]],
    m_lookup_by: dict[str, np.ndarray],
) -> dict[str, int]:
    m_tree_counts = {category: len(trees) for category, trees in m_trees.items()}

    for category in categories:
        pack_shared_memory(k_dist_thresholded_by[category], prefix, "k_dist_thresholded_by", category)
        pack_shared_memory(k_selector_by[category], prefix, "k_selector_by", category)
        pack_shared_memory(m_dist_thresholded_by[category], prefix, "m_dist_thresholded_by", category)
        pack_shared_memory(m_selector_by[category], prefix, "m_selector_by", category)
        pack_shared_memory(m_lookup_by[category], prefix, "m_lookup_by", category)
        
        for i in range(m_tree_counts[category]):
            pack_shared_rumba_tree(m_trees[category][i], prefix, "m_trees", category, str(i))

    return m_tree_counts
    
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

    k_set, k_set_dist_thresholded_by, k_selector_by, hard_match_categories = load_and_process_k(k_parquet_filename, start_year)
    m_set, m_dist_thresholded_by, m_selector_by, rumba_trees_by, m_lookup_by, invconv = load_and_process_m(
        m_parquet_filename,
        hard_match_categories,
        start_year,
        rng,
        len(k_set)
    )

    key = shared_memory_key(["find_pairs", str(os.getpid())])

    m_tree_counts = load_data_into_shared_memory(
        key,
        hard_match_categories,
        k_set_dist_thresholded_by,
        k_selector_by,
        m_dist_thresholded_by,
        m_selector_by,
        rumba_trees_by,
        m_lookup_by,
    )
    
    manager = Manager()
    queue = manager.Queue()

    logging.info("Beginning execution...")

    pool = Pool(processes=processes_count)
    pool.map_async(
        partial(
            unpack_iteration_shared_memory,
            key,
            len(k_set),
            hard_match_categories,
            m_tree_counts,
            invconv,
            queue,
        ),
        iteration_seeds,
        error_callback=lambda error: logging.error("Process error %a\r\n%a", error, error.__traceback__)
    )

    # Handle output queue
    match_storage(output_folder, k_set, m_set, queue)

    logging.info("Finished, cleaning up...")
    pool.close()
    manager.shutdown()

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
