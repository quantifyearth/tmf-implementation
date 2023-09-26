import argparse
import os
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit, float32, int32  # type: ignore
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

    s_subset_for_cov = s_subset[['elevation', 'slope', 'access', \
        'cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']]
    logging.info("Calculating covariance...")
    covarience = np.cov(s_subset_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience)

    m_distances = np.full((len(k_subset), len(s_subset)), DEFAULT_DISTANCE)

    for k_idx, k_row in k_subset.iterrows():
        # Methodology 6.5.7: find the matches.
        # There's two stages to matching - first a hard match
        # based on:
        #  * country
        #  * historic LUC
        #  * ecoregion

        if k_idx % 100 == 0 or k_idx == len(k_subset) - 1:
            logging.info(f"Calculating distances... {k_idx} of {len(k_subset)}")

        # Country is implicit in the methodology, so we don't filter
        # for it here
        filtered_s = s_subset[
            (s_subset.ecoregion == k_row.ecoregion) &
            (s_subset[luc10] == k_row[luc10]) &
            (s_subset[luc5] == k_row[luc5]) &
            (s_subset[luc0] == k_row[luc0]) &
            (s_subset.country == k_row.country)
        ]

        if len(filtered_s) == 0:
            # No matches found for this pixel, note it down and move on
            matchless.append(k_row)
            continue

        # and then a soft match based on Mahalanobis distance of
        #  * elevation
        #  * slope
        #  * accessibility
        #  * coarsened proportional coverage
        distance_columns = [
            "elevation", "slope", "access",
            "cpc0_u", "cpc0_d",
            "cpc5_u", "cpc5_d",
            "cpc10_u", "cpc10_d"
        ]
        k_soft =  np.array(k_row[distance_columns].to_list())

        just_cols_idx = filtered_s.index.to_numpy()
        just_cols = filtered_s[distance_columns].to_numpy()

        batch_distances = batch_mahalanobis(just_cols, k_soft, invconv)
        m_distances[k_idx][just_cols_idx] = batch_distances

    # Having now built up the entire Mahalanobis matrix, we now choose the minimum distance
    # for each k and s without replacement. This is a naive implementation of finding the
    # optimal solution.
    logging.info("Creating min distances...")

    # Create a mask for all entries in m_distances below DEFAULT_DISTANCE
    # This is a boolean array of the same shape as m_distances
    m_distances_mask = m_distances < DEFAULT_DISTANCE

    # Create an array of zeros with three columns and the same number of rows as m_distances_mask has True values
    # This will be used to store the k_idx, s_idx and distance for each match
    num_matches = m_distances_mask.sum()
    min_dists_idxs = np.zeros((num_matches, 2), dtype=np.int32)
    min_dists_d = np.zeros((num_matches, 1), dtype=np.float32)

    # Use the mask to fill min_dists with the k_idx, s_idx and distance for each match
    min_dists_idxs[:, 0] = np.where(m_distances_mask)[0]
    min_dists_idxs[:, 1] = np.where(m_distances_mask)[1]
    min_dists_d[:, 0] = m_distances[m_distances_mask]

    logging.info(f"Created min distances...")

    logging.info("Sorting min distances....")

    d_sorted = min_dists_d[:, 0].argsort()
    min_dists_idxs = min_dists_idxs[d_sorted]
    min_dists_d = min_dists_d[d_sorted]

    logging.info("Sorted min distances....")

    logging.info("Adding matches...")

    k_total = len(k_subset)
    s_total = len(s_subset)

    add_results, k_added = add_matches(k_total, s_total, min_dists_idxs, min_dists_d)

    logging.info(f"Got {len(add_results)} results, assembling..")

    for result in add_results:
        (k_idx, s_idx) = result
        k_row = k_subset.iloc[k_idx]
        match = s_subset.iloc[s_idx]

        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + distance_columns] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + distance_columns]
        )

    logging.info("Finished adding matches...")

    # There's a chance that didn't put every k into our results. This is
    # because the algorithm is greedy. So no we add those not matches to
    # matchless
    if len(k_added) != k_total:
        all_k = np.unique(min_dists_idxs[:,0])
        # matchless is all_k minus k_added
        matchless_k = np.setdiff1d(all_k, k_added)
        for k_idx in matchless_k:
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

# optimised batch implementation of mahalanobis distance that returns a distance per row
def batch_mahalanobis(rows, vector, invcov):
    # calculate the difference between the vector and each row (broadcasted)
    diff = rows - vector
    # calculate the distance for each row in one batch
    dists = np.sqrt((np.dot(diff, invcov) * diff).sum(axis=1))
    return dists

@jit("(int64, int64, int32[:,:], float32[:,:])", nopython=True, fastmath=True, error_model='numpy')
def add_matches(k_total, s_total, min_dists_idxs, min_dists_d):
    k_added = []
    results = []

    k_already_added = np.zeros((k_total,), dtype=np.bool_)
    s_already_added = np.zeros((s_total,), dtype=np.bool_)

    for r in range(min_dists_idxs.shape[0]):
        row = min_dists_idxs[r]

        k_idx = row[0]
        s_idx = row[1]

        if k_already_added[k_idx] or s_already_added[s_idx]:
            continue

        results.append((k_idx, s_idx))
        k_added.append(k_idx)

        k_already_added[k_idx] = True
        s_already_added[s_idx] = True

    return results, k_added

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
