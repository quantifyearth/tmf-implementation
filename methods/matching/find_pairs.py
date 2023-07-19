import os
import random
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

PROCESSES = 50

REPEAT_MATCH_FINDING = 100

def find_match_iteration(
    k_parquet_filename: str,
    s_parquet_filename: str,
    output_folder: str,
    seed: int
) -> None:
    random.seed(seed)

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

    s_subset_for_cov = s_subset[['elevation', 'slope', 'access']] # TODO CPC
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
        k_soft = [k_row.elevation, k_row.slope, k_row.access]
        filtered_s['distance'] = filtered_s.apply(
            partial(
                lambda k_row, s_row: mahalanobis(k_row, [s_row.elevation, s_row.slope, s_row.access], invconv),
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
    output_folder: str
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    random.seed(seed)
    iteration_seeds = [random.randint(0, 1000000) for _ in range(REPEAT_MATCH_FINDING)]

    with Pool(processes=PROCESSES) as pool:
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
    try:
        k_parquet_filename = sys.argv[1]
        s_parquet_filename = sys.argv[2]
        seed = int(sys.argv[3])
        output_folder = sys.argv[4]
    except (IndexError, ValueError):
        print(f"Usage: {sys.argv[0]} K_PARQUET S_PARQUET SEED_INT RESULT_FOLDER")
        sys.exit(1)

    find_pairs(
        k_parquet_filename,
        s_parquet_filename,
        seed,
        output_folder,
    )

if __name__ == "__main__":
    main()
