import argparse
import os
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit  # type: ignore
import numpy as np
import pandas as pd

from methods.common.luc import luc_matching_columns

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_match_iteration(
    k_parquet_filename: str,
    s_parquet_filename: str,
    start_year: int,
    output_folder: str,
    idx_and_seed: tuple[int, int]
) -> None:
    logging.info("Find match iteration %d of %d", idx_and_seed[0] + 1, REPEAT_MATCH_FINDING)
    random.seed(idx_and_seed[1])

    logging.info("Loading K from %s", k_parquet_filename)

    # Methodology 6.5.7: For a 10% sample of K
    k_set = pd.read_parquet(k_parquet_filename)
    k_subset = k_set.sample(
        frac=0.1,
        random_state=random.randint(0, 1000000),
    ).reset_index()

    logging.info("Loading S from %s", s_parquet_filename)
    s_set = pd.read_parquet(s_parquet_filename)

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

    logging.info("Preparing s_subset...")

    s_dist_thresholded = s_set[DISTANCE_COLUMNS] / thresholds_for_columns
    k_dist_thresholded = k_subset[DISTANCE_COLUMNS] / thresholds_for_columns

    # convert to float32 numpy arrays and make them contiguous for numba to vectorise
    s_dist_thresholded = np.ascontiguousarray(s_dist_thresholded.to_numpy(), dtype=np.float32)
    k_dist_thresholded = np.ascontiguousarray(k_dist_thresholded.to_numpy(), dtype=np.float32)

    # LUC columns are all named with the year in, so calculate the column names
    # for the years we are intested in
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    # As well as all the LUC columns for later use
    luc_columns = [x for x in s_set.columns if x.startswith('luc')]

    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]
    assert len(hard_match_columns) == HARD_COLUMN_COUNT

    # similar to the above, make the hard match columns contiguous float32 numpy arrays
    s_dist_hard = np.ascontiguousarray(s_set[hard_match_columns].to_numpy()).astype(np.int32)
    k_dist_hard = np.ascontiguousarray(k_subset[hard_match_columns].to_numpy()).astype(np.int32)

    # Methodology 6.5.5: S should be 10 times the size of K
    required = k_set.shape[0] * 10

    logging.info("Running make_s_subset_mask... required: %d", required)

    s_subset_mask = make_s_subset_mask(s_dist_thresholded, k_dist_thresholded, s_dist_hard, k_dist_hard, required)

    logging.info("Done make_s_subset_mask. s_subset_mask.shape: %s", s_subset_mask.shape)

    s_subset = s_set[s_subset_mask].reset_index()

    logging.info("Finished preparing s_subset. shape: %s", s_subset.shape)

    # Notes:
    # 1. Not all pixels may have matches
    results = []
    matchless = []

    s_subset_for_cov = s_subset[DISTANCE_COLUMNS]
    logging.info("Calculating covariance...")
    covarience = np.cov(s_subset_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience).astype(np.float32)

    # Match columns are luc10, luc5, luc0, "country" and "ecoregion"
    s_subset_match = s_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy().astype(np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    s_subset_match = np.ascontiguousarray(s_subset_match)

    # Now we do the same thing for k_subset
    k_subset_match = k_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy().astype(np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    k_subset_match = np.ascontiguousarray(k_subset_match)

    logging.info("Starting greedy matching... k_subset_match.shape: %s, s_subset_match.shape: %s",
                 k_subset_match.shape, s_subset_match.shape)

    add_results, k_idx_matchless = greedy_match(
        k_subset_match,
        s_subset_match,
        invconv
    )

    logging.info("Finished greedy matching...")

    logging.info("Starting storing matches...")

    for result in add_results:
        (k_idx, s_idx) = result
        k_row = k_subset.iloc[k_idx]
        match = s_subset.iloc[s_idx]

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

@jit(nopython=True, fastmath=True, error_model="numpy")
def make_s_subset_mask(
    s_dist_thresholded: np.ndarray,
    k_dist_thresholded: np.ndarray,
    s_dist_hard: np.ndarray,
    k_dist_hard: np.ndarray,
    number_required: int,
):
    s_include = np.zeros((s_dist_thresholded.shape[0],), dtype=np.bool_)
    # create an array that is the indexes of the rows in s_dist_thresholded and shuffle it
    s_indexes = np.arange(s_dist_thresholded.shape[0])
    np.random.shuffle(s_indexes)
    found = 0

    for position in range(s_dist_thresholded.shape[0]):
        i = s_indexes[position]
        s_row = s_dist_thresholded[i, :]
        s_hard = s_dist_hard[i]

        for k in range(k_dist_thresholded.shape[0]):
            k_row = k_dist_thresholded[k, :]
            k_hard = k_dist_hard[k]

            should_include = True

            # check that every element of s_hard matches k_hard
            hard_equals = True
            for j in range(s_hard.shape[0]):
                if s_hard[j] != k_hard[j]:
                    hard_equals = False

            if not hard_equals:
                should_include = False
            else:
                for j in range(s_row.shape[0]):
                    if abs(s_row[j] - k_row[j]) > 1.0:
                        should_include = False

            if should_include:
                s_include[i] = True
                break

        if s_include[i]:
            found += 1
            if found >= number_required:
                break

    return s_include

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
    s_subset: np.ndarray,
    invcov: np.ndarray
):
    # Create an array of booleans for rows in s still available
    s_available = np.ones((s_subset.shape[0],), dtype=np.bool_)
    total_available = s_subset.shape[0]

    results = []
    matchless = []

    s_tmp = np.zeros((s_subset.shape[0],), dtype=np.float32)

    for k_idx in range(k_subset.shape[0]):
        k_row = k_subset[k_idx, :]

        hard_matches = rows_all_true(s_subset[:, :HARD_COLUMN_COUNT] == k_row[:HARD_COLUMN_COUNT]) & s_available
        hard_matches = hard_matches.reshape(
            -1,
        )

        if total_available > 0:
            # Now calculate the distance between the k_row and all the hard matches we haven't already matched
            s_tmp[hard_matches] = batch_mahalanobis_squared(
                s_subset[hard_matches, HARD_COLUMN_COUNT:], k_row[HARD_COLUMN_COUNT:], invcov
            )
            # Find the index of the minimum distance in s_tmp[hard_matches] but map it back to the index in s_subset
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
    s_parquet_filename: str,
    start_year: int,
    seed: int,
    output_folder: str,
    processes_count: int
) -> None:
    logging.info("Starting find pairs")
    os.makedirs(output_folder, exist_ok=True)

    random.seed(seed)
    iteration_seeds = [(x, random.randint(0, 1000000)) for x in range(REPEAT_MATCH_FINDING)]

    with Pool(processes=processes_count) as pool:
        pool.map(
            partial(
                find_match_iteration,
                k_parquet_filename,
                s_parquet_filename,
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
        "--s",
        type=str,
        required=True,
        dest="s_filename",
        help="Parquet file containing pixels from S as generated by find_potential_matches.py"
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
        args.s_filename,
        args.start_year,
        args.seed,
        args.output_directory_path,
        args.processes_count
    )

if __name__ == "__main__":
    main()
