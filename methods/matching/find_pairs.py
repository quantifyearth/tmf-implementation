import argparse
import glob
import os
import logging
import re  # For extracting number from filename
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit  # type: ignore
import numpy as np
import pandas as pd
import geopandas as gpd  # Requires geopandas

from methods.common.luc import luc_matching_columns
from methods.common.geometry import area_for_geometry
from methods.common import LandUseClass

DEFAULT_DISTANCE = 10000000.0
DEBUG = False
MOLECULAR_MASS_CO2_TO_C_RATIO = 44/12

DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d"
]
HARD_COLUMN_COUNT = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_match_iteration(
    # Arguments passed via partial
    m_parquet_filename: str,  # Pass filename instead of loaded DataFrame
    start_year: int,
    evaluation_year: int,  # Added
    carbon_density: np.ndarray,  # Added
    project_area_ha: float,  # Added
    output_folder: str,
    # Arguments specific to this iteration (from pool.map)
    k_grid_filepath_and_seed: tuple[str, int]
) -> tuple[str, float]:  # Modified: Return tuple (k_grid_id, additionality_value)
    k_grid_filepath, seed = k_grid_filepath_and_seed
    k_grid_filename = os.path.basename(k_grid_filepath)

    # Extract identifier (e.g., number) from k_grid filename
    match = re.search(r'\d+', k_grid_filename)
    k_grid_id = match.group(0) if match else k_grid_filename.replace('.parquet', '')

    logging.info(f"Starting iteration for K grid: {k_grid_filename} (ID: {k_grid_id}) with seed {seed}")
    rng = np.random.default_rng(seed)

    logging.info(f"Loading K grid from {k_grid_filepath}")

    # Load the entire K grid file, no sampling
    k_subset = pd.read_parquet(k_grid_filepath).reset_index()

    logging.info(f"Loading M set from {m_parquet_filename}")
    m_set = pd.read_parquet(m_parquet_filename)
    logging.info(f"Shuffling M set for iteration {k_grid_id}...")
    m_set = m_set.sample(frac=1, random_state=rng).reset_index(drop=True)

    thresholds_for_columns = np.array([
        200.0,  # Elev
        2.5,  # Slope
        10.0,  # Access
        0.1,  # FCCs
        0.1,  # FCCs
        0.1,  # FCCs
        0.1,  # FCCs
        0.1,  # FCCs
        0.1,  # FCCs
    ])

    logging.info("Preparing s_set...")

    m_dist_thresholded_df = m_set[DISTANCE_COLUMNS] / thresholds_for_columns
    k_subset_dist_thresholded_df = k_subset[DISTANCE_COLUMNS] / thresholds_for_columns

    m_dist_thresholded = np.ascontiguousarray(m_dist_thresholded_df, dtype=np.float32)
    k_subset_dist_thresholded = np.ascontiguousarray(k_subset_dist_thresholded_df, dtype=np.float32)

    luc0, luc5, luc10 = luc_matching_columns(start_year)
    luc_columns = [x for x in m_set.columns if x.startswith('luc')]

    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]
    assert len(hard_match_columns) == HARD_COLUMN_COUNT

    m_dist_hard = np.ascontiguousarray(m_set[hard_match_columns].to_numpy()).astype(np.int32)
    k_subset_dist_hard = np.ascontiguousarray(k_subset[hard_match_columns].to_numpy()).astype(np.int32)

    required = 100

    logging.info("Running make_s_set_mask... required: %d", required)
    starting_positions = rng.integers(0, int(m_dist_thresholded.shape[0]), int(k_subset_dist_thresholded.shape[0]))
    s_set_mask_true, no_potentials = make_s_set_mask(
        m_dist_thresholded,
        k_subset_dist_thresholded,
        m_dist_hard,
        k_subset_dist_hard,
        starting_positions,
        required
    )

    logging.info("Done make_s_set_mask. s_set_mask.shape: %a", {s_set_mask_true.shape})

    s_set = m_set[s_set_mask_true]
    potentials = np.invert(no_potentials)

    k_subset = k_subset[potentials]
    logging.info("Finished preparing s_set. shape: %a", {s_set.shape})

    results = []
    matchless = []

    s_set_for_cov = s_set[DISTANCE_COLUMNS]
    logging.info("Calculating covariance...")
    covarience = np.cov(s_set_for_cov, rowvar=False)
    logging.info("Calculating inverse covariance...")
    invconv = np.linalg.inv(covarience).astype(np.float32)

    s_set_match = s_set[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    s_set_match = np.ascontiguousarray(s_set_match)

    k_subset_match = k_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    k_subset_match = np.ascontiguousarray(k_subset_match)

    logging.info("Starting greedy matching... k_subset_match.shape: %s, s_set_match.shape: %s",
                 k_subset_match.shape, s_set_match.shape)

    add_results, k_idx_matchless = greedy_match_with_shuffled_k(
        k_subset_match,
        s_set_match,
        invconv,
        rng
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
    results_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}.parquet'))

    matchless_df = pd.DataFrame(matchless, columns=k_subset.columns)
    matchless_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}_matchless.parquet'))

    logging.info(f"Calculating evaluation year ({evaluation_year}) additionality for {k_grid_id}...")
    additionality_value = 0.0
    if not results_df.empty:
        try:
            k_luc_counts = results_df[f"k_luc_{evaluation_year}"].value_counts()
            k_values = np.zeros(len(LandUseClass))
            for luc in LandUseClass:
                if k_luc_counts.get(luc.value) is not None:
                    k_values[luc.value - 1] = k_luc_counts[luc.value]
            k_proportions = k_values / len(results_df)

            s_luc_counts = results_df[f"s_luc_{evaluation_year}"].value_counts()
            s_values = np.zeros(len(LandUseClass))
            for luc in LandUseClass:
                if s_luc_counts.get(luc.value) is not None:
                    s_values[luc.value - 1] = s_luc_counts[luc.value]
            s_proportions = s_values / len(results_df)

            treatment_carbon = (k_proportions * project_area_ha * carbon_density).sum() * MOLECULAR_MASS_CO2_TO_C_RATIO
            control_carbon = (s_proportions * project_area_ha * carbon_density).sum() * MOLECULAR_MASS_CO2_TO_C_RATIO
            additionality_value = treatment_carbon - control_carbon
            logging.info(f"Iteration {k_grid_id} additionality ({evaluation_year}): {additionality_value:.4f}")
        except KeyError as e:
            logging.error(f"Could not find LUC column for evaluation year {evaluation_year}: {e}")
            additionality_value = np.nan
        except Exception as e:
            logging.error(f"Error calculating additionality for iteration {k_grid_id}: {e}")
            additionality_value = np.nan

    logging.info(f"Finished find match iteration {k_grid_id}")
    return (k_grid_id, additionality_value)

@jit(nopython=True, fastmath=True, error_model="numpy")
def make_s_set_mask(
    m_dist_thresholded: np.ndarray,
    k_subset_dist_thresholded: np.ndarray,
    m_dist_hard: np.ndarray,
    k_subset_dist_hard: np.ndarray,
    starting_positions: np.ndarray,
    required: int
):
    m_size = m_dist_thresholded.shape[0]
    k_size = k_subset_dist_thresholded.shape[0]

    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    for k in range(k_size):
        matches = 0
        k_row = k_subset_dist_thresholded[k, :]
        k_hard = k_subset_dist_hard[k]

        for index in range(m_size):
            m_index = (index + starting_positions[k]) % m_size

            m_row = m_dist_thresholded[m_index, :]
            m_hard = m_dist_hard[m_index]

            should_include = True

            hard_equals = True
            for j in range(m_hard.shape[0]):
                if m_hard[j] != k_hard[j]:
                    hard_equals = False

            if not hard_equals:
                should_include = False
            else:
                for j in range(m_row.shape[0]):
                    if abs(m_row[j] - k_row[j]) > 1.0:
                        should_include = False

            if should_include:
                s_include[m_index] = True
                matches += 1

            if matches == required:
                break

        k_miss[k] = matches == 0

    return s_include, k_miss

@jit(nopython=True, fastmath=True, error_model="numpy")
def rows_all_true(rows: np.ndarray):
    all_true = np.ones((rows.shape[0],), dtype=np.bool_)
    for row_idx in range(rows.shape[0]):
        for col_idx in range(rows.shape[1]):
            if not rows[row_idx, col_idx]:
                all_true[row_idx] = False
                break

    return all_true

@jit(nopython=True, fastmath=True, error_model="numpy")
def greedy_match_core(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray,
    k_order: np.ndarray
):
    s_available = np.ones((s_set.shape[0],), dtype=np.bool_)
    total_available = s_set.shape[0]

    results = []
    matchless = []

    s_tmp = np.zeros((s_set.shape[0],), dtype=np.float32)

    for k_idx_original in k_order:
        k_row = k_subset[k_idx_original, :]

        hard_matches = rows_all_true(s_set[:, :HARD_COLUMN_COUNT] == k_row[:HARD_COLUMN_COUNT]) & s_available
        hard_matches = hard_matches.reshape(-1)

        if total_available > 0:
            s_tmp[hard_matches] = batch_mahalanobis_squared(
                s_set[hard_matches, HARD_COLUMN_COUNT:], k_row[HARD_COLUMN_COUNT:], invcov
            )
            if np.any(hard_matches):
                min_dist_idx = np.argmin(s_tmp[hard_matches])
                s_idx = np.arange(s_tmp.shape[0])[hard_matches][min_dist_idx]

                results.append((k_idx_original, s_idx))
                s_available[s_idx] = False
                total_available -= 1
            else:
                matchless.append(k_idx_original)
        else:
            matchless.append(k_idx_original)

    return (results, matchless)

def greedy_match_with_shuffled_k(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray,
    rng: np.random.Generator
):
    k_indices = np.arange(k_subset.shape[0])
    rng.shuffle(k_indices)
    return greedy_match_core(k_subset, s_set, invcov, k_indices)

@jit(nopython=True, fastmath=True, error_model="numpy")
def batch_mahalanobis_squared(rows, vector, invcov):
    diff = rows - vector
    dists = (np.dot(diff, invcov) * diff).sum(axis=1)
    return dists

def calculate_rse(estimates: list[float]) -> float:
    valid_estimates = [e for e in estimates if not np.isnan(e)]
    n = len(valid_estimates)
    if n < 2:
        return np.inf

    mean_val = np.mean(valid_estimates)
    if abs(mean_val) < 1e-9:
        return np.inf

    std_dev = np.std(valid_estimates, ddof=1)
    std_err = std_dev / np.sqrt(n)
    rse = std_err / abs(mean_val)
    return rse

# --- Helper function to extract number for sorting ---
def extract_k_grid_number(filepath: str) -> int:
    """Extracts the integer part from filenames like 'k_123.parquet'."""
    basename = os.path.basename(filepath)
    match = re.search(r'k_(\d+)\.parquet$', basename)
    if match:
        return int(match.group(1))
    else:
        # Fallback for unexpected filenames, sort them last
        logging.warning(f"Could not extract grid number from {basename}. Placing it at the end.")
        return float('inf')

def find_pairs(
    k_directory: str,
    m_parquet_filename: str,
    start_year: int,
    evaluation_year: int,
    carbon_density: np.ndarray,
    project_area_ha: float,
    seed: int,
    output_folder: str,
    batch_size: int,
    rse_threshold: float,
    processes_count: int
) -> None:
    logging.info("Starting find pairs")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(m_parquet_filename):
        logging.error(f"M set file not found: {m_parquet_filename}")
        return

    # Find K grid files
    k_grid_files_unsorted = glob.glob(os.path.join(k_directory, "k_*.parquet"))
    if not k_grid_files_unsorted:
        logging.error(f"No k_*.parquet files found in directory: {k_directory}")
        return

    # Sort K grid files numerically using the helper function
    k_grid_files = sorted(k_grid_files_unsorted, key=extract_k_grid_number)

    num_k_grids = len(k_grid_files)
    logging.info(f"Found and numerically sorted {num_k_grids} K grid files to process.")

    rng = np.random.default_rng(seed)
    iteration_seeds = rng.integers(0, 1000000, num_k_grids)
    map_arguments = list(zip(k_grid_files, iteration_seeds))

    all_additionality_estimates = []
    processed_k_grid_ids = []
    total_processed = 0

    iteration_func = partial(
        find_match_iteration,
        m_parquet_filename,
        start_year,
        evaluation_year,
        carbon_density,
        project_area_ha,
        output_folder
    )

    with Pool(processes=processes_count) as pool:
        for i in range(0, num_k_grids, batch_size):
            batch_args = map_arguments[i:min(i + batch_size, num_k_grids)]
            if not batch_args:
                break

            logging.info(f"Processing batch {i//batch_size + 1}/{ (num_k_grids + batch_size - 1)//batch_size } (K grids {extract_k_grid_number(batch_args[0][0])} to {extract_k_grid_number(batch_args[-1][0])})...") # Log numerical range
            batch_results = pool.map(iteration_func, batch_args)

            for k_grid_id, additionality_value in batch_results:
                processed_k_grid_ids.append(k_grid_id)
                all_additionality_estimates.append(additionality_value)

            total_processed = len(all_additionality_estimates)
            current_rse = calculate_rse(all_additionality_estimates)
            logging.info(f"Processed {total_processed}/{num_k_grids} K grids. Current RSE: {current_rse:.4f} (Threshold: {rse_threshold:.4f})")

            if current_rse <= rse_threshold:
                logging.info(f"Convergence reached (RSE <= {rse_threshold}). Stopping.")
                break

    logging.info(f"Finished processing. Total K grids processed: {total_processed}")

def main():
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Takes K and S and finds 100 sets of matches.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_directory",
        help="Directory containing K set Parquet files (k_*.parquet)"
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
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Evaluation year for intermediate additionality calculation."
    )
    parser.add_argument(
        "--density",
        type=str,
        required=True,
        dest="carbon_density_file",
        help="Path to the carbon density CSV or Parquet file."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_file",
        help="GeoJSON file containing the project boundary (for area calculation)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        dest="batch_size",
        help="Number of K grids to process per batch before checking convergence."
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
        "--rse_threshold",
        type=float,
        default=0.10,
        dest="rse_threshold",
        help="Relative Standard Error threshold to stop processing batches (e.g., 0.10 for 10%)."
    )
    parser.add_argument(
        "--j",
        type=int,
        required=False,
        default=round(cpu_count() / 8),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    logging.info(f"Loading carbon density from {args.carbon_density_file}")
    _, ext = os.path.splitext(args.carbon_density_file)
    if ext == ".csv":
        density_df = pd.read_csv(args.carbon_density_file)
    elif ext == ".parquet":
        density_df = pd.read_parquet(args.carbon_density_file)
    else:
        logging.error(f"Unrecognised file extension for density file: {ext}")
        exit(1)
    density = np.zeros(len(LandUseClass))
    for _, row in density_df.iterrows():
        try:
            luc_val = int(row["land use class"])
            if 1 <= luc_val <= len(LandUseClass):
                density[luc_val - 1] = row["carbon density"]
        except (ValueError, KeyError):
            logging.warning(f"Skipping invalid row in density file: {row}")

    logging.info(f"Loading project boundary from {args.project_boundary_file}")
    project_gpd = gpd.read_file(args.project_boundary_file)
    project_area_msq = area_for_geometry(project_gpd)
    project_area_ha = project_area_msq / 10000.0
    logging.info(f"Calculated project area: {project_area_ha:.2f} ha")

    find_pairs(
        args.k_directory,
        args.m_filename,
        args.start_year,
        args.evaluation_year,
        density,
        project_area_ha,
        args.seed,
        args.output_directory_path,
        args.batch_size,
        args.rse_threshold,
        args.processes_count
    )

if __name__ == "__main__":
    main()
