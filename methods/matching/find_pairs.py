import argparse
import os
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis  # type: ignore

from methods.common.luc import luc_matching_columns

REPEAT_MATCH_FINDING = 100

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

    # Methodology 6.5.7: For a 10% sample of K
    k_set = pd.read_parquet(k_parquet_filename)
    k_subset = k_set.sample(
        frac=0.1,
        random_state=random.randint(0, 1000000),
    )

    # Methodology 6.5.5: S should be 10 times the size of K
    s_set = pd.read_parquet(s_parquet_filename)
    s_subset = s_set.sample(
        n=k_set.shape[0] * 10,
        random_state=random.randint(0, 1000000),
    )

    # in the current methodology version (1.1), it's possible for
    # multiple pixels in k to map to the same pixel in S
    results = []

    # LUC columns are all named with the year in, so calculate the column names
    # for the years we are intested in
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    # As well as all the LUC columns for later use
    luc_columns = [x for x in s_set.columns if x.startswith('luc')]

    s_subset_for_cov = s_subset[['elevation', 'slope', 'access', \
        'cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']]
    covarience = np.cov(s_subset_for_cov, rowvar=False)
    invconv = np.linalg.inv(covarience)

    for _, k_row in k_subset.iterrows():
        # Methodology 6.5.7: find the matches.
        # There's two stages to matching - first a hard match
        # based on:
        #  * country
        #  * historic LUC
        #  * ecoregion

        # Country is implicit in the methodology, so we don't filter
        # for it here
        filtered_s = s_subset[
            (s_subset.ecoregion == k_row.ecoregion) &
            (s_subset[luc10] == k_row[luc10]) &
            (s_subset[luc5] == k_row[luc5]) &
            (s_subset[luc0] == k_row[luc0])
        ]

        if len(filtered_s) == 0:
            # No matches found for this pixel, move on
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
        just_cols = filtered_s[distance_columns].to_numpy()

        min_distance = 10000000000.0
        min_index = None
        for index in range(len(just_cols)): # pylint: disable=C0200
            s_row = just_cols[index]
            distance = mahalanobis(k_soft, s_row, invconv)
            if distance < min_distance:
                min_distance = distance
                min_index = index
        if min_index is None:
            logging.warning("We got no minimum despite having %d potential matches", len(filtered_s))
            continue
        match = filtered_s.iloc[min_index]

        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + distance_columns] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + distance_columns]
        )

    columns = ['k_lat', 'k_lng'] + \
        [f'k_{x}' for x in luc_columns + distance_columns] + \
        ['s_lat', 's_lng'] + \
        [f's_{x}' for x in luc_columns + distance_columns]

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}.parquet'))


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
