import argparse
import os
import logging
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import njit  # type: ignore
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import faiss

from methods.common.luc import luc_matching_columns

REPEAT_MATCH_FINDING = 100
DEBUG = False

K_SUB_PROPORTION = 0.01
M_SUB_PROPORTION = 1
# Number of clusters
NUM_CLUSTERS = 9
# Number of iterations for K means fitting
NUM_ITERATIONS = 100
RELATIVE_MATCH_YEARS = [-10, -5, 0]


DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "cpc0_u", "cpc0_d",
    "cpc5_u", "cpc5_d",
    "cpc10_u", "cpc10_d"
]
HARD_COLUMN_COUNT = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# this function uses loops instead of numpy vector operations
@njit
def loop_match(m_pca, k_pca):
    picked = np.ones((m_pca.shape[0],), dtype=np.bool_)
    fast_matches = np.full(k_pca.shape[0], -1, dtype=np.int32)
    for i in range(0, k_pca.shape[0]):
        min_squared_diff_sum = np.inf
        min_squared_diff_j = -1
        for j in range(0, m_pca.shape[0]):
            if picked[j]:
                squared_diff_sum = np.sum((m_pca[j, :] - k_pca[i, :])**2)
                if squared_diff_sum < min_squared_diff_sum:
                    min_squared_diff_sum = squared_diff_sum
                    min_squared_diff_j = j
        fast_matches[i] = min_squared_diff_j
        picked[min_squared_diff_j] = False
    return fast_matches

### Now try with real numbers

def to_int32(x):
    # Normalize the data to the range 0 to 1
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_data = (x - min_val) / (max_val - min_val)
    # Scale the normalized data to the range 0 to 255 for unsigned 8-bit integers
    scaled_data = normalized_data * 255
    # Convert to 32-bit integers (0 to 255)
    int32_data = scaled_data.astype(np.int32)
    return int32_data

def to_pca_int32(x):
    # Perform PCA and convert to dataframe
    pca = PCA(n_components=min(len(x), len(x.columns)), 
            whiten=False)  # Centering and scaling done by default
    pca_result = pca.fit_transform(x)
    pca_df = pd.DataFrame(pca_result)
    # Convert all columns to int8
    pca_32 = pca_df.apply(to_int32)
    return pca_32


def calculate_smd(group1, group2):
    # Means
    mean1, mean2 = np.mean(group1), np.mean(group2)
    # Standard deviations
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    # Sample sizes
    n1, n2 = len(group1), len(group2)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    # Standardized mean difference
    smd = (mean1 - mean2) / pooled_std
    return smd, mean1, mean2, pooled_std

def rename_luc_columns(df, start_year):

    # Define the range of years based on the central start_year
    years = range(start_year - 10, start_year + 11)  # Adjust the range as needed
    new_column_names = {f'luc_{year}': f'luc_{year - start_year}' for year in years}
    
    # Rename columns based on the new column names mapping
    renamed_df = df.rename(columns=new_column_names)
    
    return renamed_df

def find_match_iteration(
    k_pixels: pd.DataFrame,
    m_pixels: pd.DataFrame,
    start_year: int,
    luc_match: bool,
    output_folder: str,
    idx_and_seed: tuple[int, int]
) -> None:
    logging.info("Find match iteration %d of %d", idx_and_seed[0] + 1, REPEAT_MATCH_FINDING)
    rng = np.random.default_rng(idx_and_seed[1])
    
    match_years = [start_year + year for year in RELATIVE_MATCH_YEARS]
    # The categorical columns:
    if luc_match:
        match_cats = ["ecoregion", "country", "cluster"] + ["luc_-10", "luc_-5", "luc_0"]
    else:
        match_cats = ["ecoregion", "country", "cluster"]
    
    if(m_pixels.shape[0] > (k_pixels.shape[0])):
        m_sub_size = int(k_pixels.shape[0]) # First down sample M as it is ~230 million points    
        m_random_indices = np.random.choice(m_pixels.shape[0], size=m_sub_size, replace=False)
        m_pixels = m_pixels.iloc[m_random_indices]
    
    # rename columns of each 
    k_pixels_renamed = rename_luc_columns(k_pixels, start_year)
    m_pixels_renamed = rename_luc_columns(m_pixels, start_year-10)
        
    # concat m and k
    km_pixels = pd.concat([k_pixels_renamed.assign(trt='trt', ID=range(0, len(k_pixels))),
                          m_pixels_renamed.assign(trt='ctrl', 
                                                ID=range(0, len(m_pixels)))], 
                                                ignore_index=True)
    
    # Extract only the continuous columns
    km_pixels_distance = km_pixels[DISTANCE_COLUMNS]
    # PCA transform and conversion to 32 bit ints
    logging.info("Transforming continuous variables to PCA space")
    km_pca = to_pca_int32(km_pixels_distance)
    # Find clusters using Kmeans
    logging.info("Starting cluster assignment using kmeans")
    # Initialize the KMeans object
    kmeans = faiss.Kmeans(d=km_pca.shape[1], k=NUM_CLUSTERS, niter=NUM_ITERATIONS, verbose=True)
    # Perform clustering
    kmeans.train(km_pca)
    # Get cluster assignments
    km_pixels['cluster'] = kmeans.index.search(km_pca, 1)[1].flatten()
    
    # Extract K and M pixels 
    k_pixels = km_pixels.loc[km_pixels['trt'] == 'trt']
    m_pixels = km_pixels.loc[km_pixels['trt'] == 'ctrl']
    # Extract K and M PCA transforms
    k_pca = km_pca.loc[km_pixels['trt'] == 'trt'].to_numpy()
    m_pca = km_pca.loc[km_pixels['trt'] == 'ctrl'].to_numpy()
    # Draw subsamples
    # Methodology 6.5.7: Needs to be updated !!!
    k_sub_size = int(k_pixels.shape[0]* K_SUB_PROPORTION)
    m_sub_size = int(m_pixels.shape[0] * M_SUB_PROPORTION)
    # Define indexs for the samples from K and M
    k_random_indices = np.random.choice(k_pixels.shape[0], size=k_sub_size, replace=False)
    m_random_indices = np.random.choice(m_pixels.shape[0], size=m_sub_size, replace=False)
    # Take random samples from K and M pixels
    k_sub = k_pixels.iloc[k_random_indices]
    m_sub = m_pixels.iloc[m_random_indices]
    # Take corresponding random samples from the PCA transformed K and M 
    k_sub_pca = k_pca[k_random_indices,:]
    m_sub_pca = m_pca[m_random_indices,:]
    
    logging.info("Starting greedy matching... k_sub.shape: %s, m_sub.shape: %s",
                 k_sub.shape, m_sub.shape)
    
    pairs, matchless = greedy_match(
        k_sub,
        m_sub,
        k_sub_pca,
        m_sub_pca,
        match_cats
    )
    
    # Combine all the pairs DataFrames in the list into a single DataFrame
    combined_pairs = pd.concat(pairs, ignore_index=True)
    # Combine all the matchess DataFrames in the list into a single DataFrame
    combined_matchless = pd.concat(matchless, ignore_index=True)
    logging.info("Finished greedy matching...")
    
    logging.info("Starting storing matches...")
    combined_pairs_df = pd.DataFrame(combined_pairs)
    combined_pairs_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}.parquet'))
    
    combined_matchless_df = pd.DataFrame(combined_matchless)
    combined_matchless_df.to_parquet(os.path.join(output_folder, f'{idx_and_seed[1]}_matchless.parquet'))
    
    logging.info("Finished find match iteration")

def greedy_match(
    k_sub: pd.DataFrame,
    m_sub: pd.DataFrame,
    k_sub_pca: np.ndarray,
    m_sub_pca: np.ndarray,
    match_cats: list
):
    # Identify the unique combinations of categorical columns
    k_cat_combinations = k_sub[match_cats].drop_duplicates().sort_values(by=match_cats, ascending=[True] * len(match_cats))
    
    # Not all pixels may have matches
    pairs = []
    matchless = []
    
    for i in range(0, k_cat_combinations.shape[0]):
        # i = 6 # ith element of the unique combinations of the luc time series in k
        # for in range()
        k_cat_comb = k_cat_combinations.iloc[i]
        k_cat_index = k_sub[match_cats] == k_cat_comb
        k_cat = k_sub[(k_cat_index).all(axis=1)]
        k_cat_pca = k_sub_pca[(k_cat_index).all(axis=1)]
        
        # Find the subset in km_pixels that matches this combination
        m_cat_index = m_sub[match_cats] == k_cat_comb
        m_cat = m_sub[(m_cat_index).all(axis=1)]
        m_cat_pca = m_sub_pca[(m_cat_index).all(axis=1)]
        
        # If there is no suitable match for the pre-project luc time series
        # Then it may be preferable to just take the luc state at t0
        # m_luc_comb = m_pixels[(m_pixels[match_luc_years[1:3]] == K_luc_comb[1:3]).all(axis=1)]
        # m_luc_comb = m_pixels[(m_pixels[match_luc_years[2:3]] == K_luc_comb[2:3]).all(axis=1)]
        # For now if there are no matches return nothing
        
        if(m_cat.shape[0] < k_cat.shape[0] * 5):
            matchless.append(k_cat)
            continue
        
        matches_index = loop_match(m_cat_pca, k_cat_pca)
        m_cat_matches = m_cat.iloc[matches_index]
        
        # Join the pairs into one dataframe:
        k_cat = k_cat.reset_index(drop = True)
        m_cat_matches = m_cat_matches.reset_index(drop = True)  
        pairs_df = pd.concat([k_cat.add_prefix('k_'), m_cat_matches.add_prefix('s_')], axis=1)
        # Append the resulting DataFrame to the list
        pairs.append(pairs_df)
    
    return (pairs, matchless)

def find_pairs(
    k_parquet_filename: str,
    m_parquet_filename: str,
    start_year: int,
    luc_match: bool,
    seed: int,
    output_folder: str,
    processes_count: int
) -> None:
    logging.info("Loading K from %s", k_parquet_filename)
    k_pixels = pd.read_parquet(k_parquet_filename)
    logging.info("Loading M from %s", m_parquet_filename)
    m_pixels = pd.read_parquet(m_parquet_filename)
    
    logging.info("Starting find pairs")
    os.makedirs(output_folder, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    iteration_seeds = zip(range(REPEAT_MATCH_FINDING), rng.integers(0, 1000000, REPEAT_MATCH_FINDING))
    
    with Pool(processes=processes_count) as pool:
        pool.map(
            partial(
                find_match_iteration,
                k_pixels,
                m_pixels,
                start_year,
                luc_match,
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
        "--luc_match",
        type=bool,
        required=True,
        dest="luc_match",
        help="Boolean determines whether matching should include LUCs."
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
        args.luc_match,
        args.seed,
        args.output_directory_path,
        args.processes_count
    )

if __name__ == "__main__":
    main()
