import argparse
import os
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis  # type: ignore

REPEAT_MATCH_FINDING = 100

def find_match_iteration(
    k_parquet_filename: str,
    s_parquet_filename: str,
    output_folder: str,
    idx_and_seed: tuple[int, int]
) -> None:
    logging.info(f"Find match iteration {idx_and_seed[0] + 1} of {REPEAT_MATCH_FINDING}")
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

    # These are named with years in, but are currently always in the
    # order of t-10, t-5, t, t+1, t+2, ... where t is project start
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
            (s_subset[luc_columns[0]] == k_row.luc10) &
            (s_subset[luc_columns[1]] == k_row.luc5) &
            (s_subset[luc_columns[2]] == k_row.luc0)
        ].copy()

        # and then a soft match based on Mahalanobis distance of
        #  * elevation
        #  * slope
        #  * accessibility
        #  * coarsened proportional coverage
        k_soft = [k_row.elevation, k_row.slope, k_row.access, k_row.cpc0_u,
            k_row.cpc0_d, k_row.cpc5_u, k_row.cpc5_d, k_row.cpc10_u, k_row.cpc10_d]
        filtered_s['distance'] = filtered_s.apply(
            partial(
                lambda k_row, s_row: mahalanobis(k_row, [s_row.elevation, s_row.slope, s_row.access,
                    s_row.cpc0_u, s_row.cpc0_d, s_row.cpc5_u, s_row.cpc5_d, s_row.cpc10_u, s_row.cpc10_d],
                    invconv),
                k_soft
            ),
            axis=1
        )
        minimal_s = filtered_s[filtered_s['distance']==filtered_s['distance'].min()]
        match = minimal_s.iloc[0]

        results.append([
            k_row.lat,
            k_row.lng,
            match.lat,
            match.lng
        ] + [match[x] for x in luc_columns[2:]])

    results_df = pd.DataFrame(results, columns=['k_lat', 'k_lng', 's_lat', 's_lng'] + luc_columns[2:])
    results_df.to_parquet(os.path.join(output_folder, f'{seed}.parquet'))


def find_pairs(
    k_parquet_filename: str,
    s_parquet_filename: str,
    seed: int,
    output_folder: str,
    processes_count: int
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    random.seed(seed)
    iteration_seeds = [(x, random.randint(0, 1000000)) for x in range(REPEAT_MATCH_FINDING)]

    with Pool(processes=processes_count) as pool:
        pool.map(
            partial(
                find_match_iteration,
                k_parquet_filename,
                s_parquet_filename,
                output_folder
            ),
            iteration_seeds
        )

def main():
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
        default=round(cpu_count() / 3),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    find_pairs(
        args.k_filename,
        args.s_filename,
        args.seed,
        args.output_directory_path,
        args.processes_count
    )

if __name__ == "__main__":
    main()
