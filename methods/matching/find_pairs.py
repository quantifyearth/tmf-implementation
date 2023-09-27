import argparse
import os
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit, float32, int32, gdb  # type: ignore
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis  # type: ignore

from methods.common.luc import luc_matching_columns

REPEAT_MATCH_FINDING = 100
DEFAULT_DISTANCE = 10000000.0

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

    logging.info(f"Loading K from {k_parquet_filename}")

    # Methodology 6.5.7: For a 10% sample of K
    k_set = pd.read_parquet(k_parquet_filename)
    k_subset = k_set.sample(
        frac=0.1,
        random_state=random.randint(0, 1000000),
    ).reset_index()

    logging.info(f"Loading S from {s_parquet_filename}")
    # Methodology 6.5.5: S should be 10 times the size of K
    s_set = pd.read_parquet(s_parquet_filename)
    s_subset = s_set.sample(
        n=k_set.shape[0] * 10,
        random_state=random.randint(0, 1000000),
    ).reset_index()

    # Notes:
    # 1. in the current methodology version (1.1), it's possible for
    # multiple pixels in k to map to the same pixel in S
    # 2. Not all pixels may have matches
    results = []
    matchless = []

    # LUC columns are all named with the year in, so calculate the column names
    # for the years we are intested in
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    # As well as all the LUC columns for later use
    luc_columns = [x for x in s_set.columns if x.startswith('luc')]

    distance_columns = [
        "elevation", "slope", "access",
        "cpc0_u", "cpc0_d",
        "cpc5_u", "cpc5_d",
        "cpc10_u", "cpc10_d"
    ]

    s_subset_for_cov = s_subset[distance_columns]
    logging.info("Calculating covariance...")
    covarience = np.cov(s_subset_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience).astype(np.float32)

    m_distances = np.full((len(k_subset), len(s_subset)), DEFAULT_DISTANCE)

    # Match columns are luc10, luc5, luc0, "country" and "ecoregion"
    s_subset_match = s_subset[['country', 'ecoregion', luc10, luc5, luc0] + distance_columns].to_numpy().astype(np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    s_subset_match = np.ascontiguousarray(s_subset_match)

    # Now we do the same thing for k_subset
    k_subset_match = k_subset[['country', 'ecoregion', luc10, luc5, luc0] + distance_columns].to_numpy().astype(np.float32)
    # this is required so numba can vectorise the loop in greedy_match
    k_subset_match = np.ascontiguousarray(k_subset_match)

    logging.info("Starting greedy matching...")

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

        if k_row["country"] != match["country"]:
            print(k_row)
            print(match)
            raise ValueError("Nah!!!!")

        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + distance_columns] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + distance_columns]
        )

    logging.info("Finished storing matches...")

    for k_idx in k_idx_matchless:
        k_row = k_subset.iloc[k_idx]
        matchless.append(k_row)

    columns = ['k_lat', 'k_lng'] + \
        [f'k_{x}' for x in luc_columns + distance_columns] + \
        ['s_lat', 's_lng'] + \
        [f's_{x}' for x in luc_columns + distance_columns]

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}.parquet'))

    matchless_df = pd.DataFrame(matchless, columns=k_set.columns)
    matchless_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}_matchless.parquet'))

    logging.info("Finished find match iteration")

# Function which returns a boolean array indicating whether all values in a row are true
@jit(nopython=True, fastmath=True, error_model="numpy", cache=True)
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

@jit(nopython=True, fastmath=True, error_model="numpy", cache=True)
def greedy_match(
    k_subset: np.ndarray,
    s_subset: np.ndarray,
    invcov: np.ndarray
):
    # Create an array of booleans for rows in s still available
    s_available = np.ones((s_subset.shape[0],), dtype=np.bool_)

    results = []
    matchless = []

    s_tmp = np.full((s_subset.shape[0],), dtype=np.float32, fill_value=100000.0)

    for k_idx in range(k_subset.shape[0]):
        k_row = k_subset[k_idx, :]

        # Find all rows in s_subset that match on the first 5 columns
        hard_matches = rows_all_true(s_subset[:, :5] == k_row[:5]) & s_available
        hard_matches = hard_matches.reshape(-1,)

        # Now calculate the distance between the k_row and all the hard matches we haven't already matched
        s_tmp[hard_matches] = batch_mahalanobis_squared(s_subset[hard_matches, 5:], k_row[5:], invcov)

        # Find the minimum distance if there are any hard matches
        if np.any(hard_matches):
            min_dist = np.min(s_tmp[hard_matches])
            # Find the index of the minimum distance (in s_subset)
            min_dist_idx = np.argmin(s_tmp)

            results.append((k_idx, min_dist_idx))
            s_available[min_dist_idx] = False
        else:
            matchless.append(k_idx)

    return (results, matchless)

# optimised batch implementation of mahalanobis distance that returns a distance per row
@jit(nopython=True, fastmath=True, error_model="numpy", cache=True)
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
