#!/usr/bin/env python3

import argparse
import glob
import os
import logging
import re
import gc
import datetime
import psutil
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method
from shapely.geometry import shape
from pyproj import Geod
from numba import jit
from methods.common.luc import luc_matching_columns
from methods.common import LandUseClass

# Constants
HARD_COLUMN_COUNT = 5  # Number of categorical variables requiring exact match
MOLECULAR_MASS_CO2_TO_C_RATIO = 44.01 / 12.01  # CO2 to C conversion ratio

# Continuous covariates used in Euclidean distance
DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d"
]

DEBUG = False
MEMORY_LOG_FILE = None

def load_carbon_density(path):
    """Load carbon density mapping from CSV file."""
    df = pd.read_csv(path)
    # expects columns "land use class" and "carbon density"
    return dict(zip(df["land use class"], df["carbon density"]))

def compute_project_area_ha(geojson_path):
    """
    Compute project area in hectares from a GeoJSON file (WGS84),
    using geodetic polygon areas via pyproj.Geod.
    """
    geod = Geod(ellps="WGS84")
    total_m2 = 0.0

    # Load the GeoJSON
    with open(geojson_path, 'r') as fp:
        gj = json.load(fp)

    # Support FeatureCollection or single Feature
    feats = gj.get("features") or [gj]
    for feat in feats:
        geom = shape(feat["geometry"])
        # Polygon
        if geom.geom_type == "Polygon":
            total_m2 += _poly_area(geom, geod)
        # MultiPolygon
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                total_m2 += _poly_area(poly, geod)

    # convert m² → hectares
    return total_m2 / 10_000.0

def _poly_area(poly, geod):
    """Compute the geodetic area of a single shapely Polygon (m²)."""
    # exterior ring
    lon, lat = zip(*poly.exterior.coords)
    area_ext = abs(geod.polygon_area_perimeter(lon, lat)[0])
    # subtract any holes
    for interior in poly.interiors:
        lon_i, lat_i = zip(*interior.coords)
        area_ext -= abs(geod.polygon_area_perimeter(lon_i, lat_i)[0])
    return area_ext

def build_match_key(df, start_year):
    """Build matching key for grouping pixels with same characteristics."""
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    return (
        df["ecoregion"].astype(int).astype(str)
        + "|" + df["country"].astype(int).astype(str)
        + "|" + df[luc0].astype(int).astype(str)
        + "|" + df[luc5].astype(int).astype(str)
        + "|" + df[luc10].astype(int).astype(str)
    )

def setup_memory_logging(output_folder):
    """Setup memory logging to file in output directory."""
    global MEMORY_LOG_FILE
    MEMORY_LOG_FILE = os.path.join(output_folder, "memory_usage_log.txt")
    
    # Initialize log file with header
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(MEMORY_LOG_FILE, 'w') as f:
        f.write(f"Memory Usage Log - Started at {timestamp}\n")
        f.write("="*50 + "\n\n")

def log_memory_usage(label="", detailed=False):
    """Log memory usage of the process to both console and file."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        log_msg = f"Memory {label}: RSS {rss_mb:.1f} MB"
        logging.info(log_msg)
        
        if detailed:
            detailed_msg = f"Memory {label}: RSS {rss_mb:.1f} MB, VMS {vms_mb:.1f} MB"
            logging.info(detailed_msg)
        
        # Write to memory log file
        if MEMORY_LOG_FILE:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            with open(MEMORY_LOG_FILE, 'a') as f:
                f.write(f"[{timestamp}] PID {os.getpid()}: {log_msg}\n")
                
        return rss_mb
    except:
        return 0

def log_memory_change(label_before, label_after, memory_before=None):
    """Log memory change between two points."""
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    if memory_before is not None:
        change = memory_after - memory_before
        sign = "+" if change >= 0 else ""
        change_msg = f"Memory {label_after}: {memory_after:.1f} MB ({sign}{change:.1f} MB from {label_before})"
    else:
        change_msg = f"Memory {label_after}: {memory_after:.1f} MB"
        
    logging.info(change_msg)
    
    # Write to memory log file
    if MEMORY_LOG_FILE:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] PID {os.getpid()}: {change_msg}\n")
    
    return memory_after

def log_dataframe_memory(df, df_name):
    """Log memory usage of a DataFrame."""
    total_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    shape_info = f"{df.shape[0]:,} rows × {df.shape[1]} cols"
    memory_msg = f"{df_name} DataFrame: {total_memory:.1f} MB ({shape_info})"
    
    logging.info(memory_msg)
    
    # Write to memory log file
    if MEMORY_LOG_FILE:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] PID {os.getpid()}: {memory_msg}\n")

def log_array_memory(arr, arr_name):
    """Log memory usage of a numpy array."""
    if hasattr(arr, 'nbytes'):
        memory_mb = arr.nbytes / 1024 / 1024
        if memory_mb > 1:  # Only log arrays > 1MB
            memory_msg = f"{arr_name} array: {memory_mb:.1f} MB ({arr.shape})"
            logging.info(memory_msg)
            
            if MEMORY_LOG_FILE:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                with open(MEMORY_LOG_FILE, 'a') as f:
                    f.write(f"[{timestamp}] PID {os.getpid()}: {memory_msg}\n")

def log_system_memory():
    """Log overall system memory usage."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        system_msg = (f"System Memory: {memory.used / 1024**3:.1f}GB used / "
                     f"{memory.total / 1024**3:.1f}GB total ({memory.percent:.1f}%), "
                     f"Swap: {swap.used / 1024**3:.1f}GB / {swap.total / 1024**3:.1f}GB")
        
        logging.info(system_msg)
        
        # Write to memory log file
        if MEMORY_LOG_FILE:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            with open(MEMORY_LOG_FILE, 'a') as f:
                f.write(f"[{timestamp}] SYSTEM: {system_msg}\n")
    except:
        pass

def find_match_iteration(
    m_sample_filename: str,
    start_year: int,
    evaluation_year: int,
    carbon_density: np.ndarray,
    project_area_ha: float,
    output_folder: str,
    k_grid_filepath_and_seed: tuple,
    shuffle_seed: int
) -> tuple:
    """
    Process a single K grid file to find matching S pixels from the M set.
    
    Returns:
        Tuple of K grid ID and calculated additionality value
    """
    k_grid_filepath, seed = k_grid_filepath_and_seed
    k_grid_filename = os.path.basename(k_grid_filepath)

    # Extract identifier from k_grid filename
    match = re.search(r'\d+', k_grid_filename)
    k_grid_id = match.group(0) if match else k_grid_filename.replace('.parquet', '')

    logging.info(f"Starting iteration for K grid: {k_grid_filename} (ID: {k_grid_id}) with seed {seed}")
    log_memory_usage(f"iteration start for {k_grid_id}", detailed=True)
    
    rng = np.random.default_rng(seed)

    # Load K grid
    logging.info(f"Loading K grid from {k_grid_filepath}")
    k_subset = pd.read_parquet(k_grid_filepath).reset_index(drop=True)
    log_dataframe_memory(k_subset, "K grid")

    # Load pre-sampled M set
    logging.info(f"Loading pre-sampled M set from {m_sample_filename}")
    m_set = pd.read_parquet(m_sample_filename)

    # No shuffling or sampling of M set – use it as is
    
    # Thresholds for normalising continuous variables
    thresholds_for_columns = np.array([
        200.0,  # Elevation (metres)
        2.5,    # Slope (degrees)
        10.0,   # Accessibility (cost units)
        0.1,    # Forest cover change metrics
        0.1,    # FCCs are normalised with smaller thresholds
        0.1,    # as they're already on a 0-1 scale
        0.1,
        0.1,
        0.1,
    ])

    # Normalise continuous variables by dividing by thresholds
    m_dist_thresholded_df = m_set[DISTANCE_COLUMNS] / thresholds_for_columns
    k_subset_dist_thresholded_df = k_subset[DISTANCE_COLUMNS] / thresholds_for_columns

    # Convert to contiguous arrays for faster numba processing
    m_dist_thresholded = np.ascontiguousarray(m_dist_thresholded_df, dtype=np.float32)
    k_subset_dist_thresholded = np.ascontiguousarray(k_subset_dist_thresholded_df, dtype=np.float32)

    # Get land use class column names for exact matching
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    luc_columns = [x for x in m_set.columns if x.startswith('luc')]

    # Columns requiring exact matches (categorical variables)
    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]

    # Prepare categorical variables as integer arrays
    m_dist_hard = np.ascontiguousarray(m_set[hard_match_columns].to_numpy()).astype(np.int32)
    k_subset_dist_hard = np.ascontiguousarray(k_subset[hard_match_columns].to_numpy()).astype(np.int32)

    # Free intermediate dataframes
    del m_dist_thresholded_df, k_subset_dist_thresholded_df
    gc.collect()

    # Max number of potential matches to find per K pixel
    max_potential_matches = 100

    logging.info("Running make_s_set_mask...")
    # Create random starting positions to avoid bias in candidate selection
    starting_positions = rng.integers(0, int(m_dist_thresholded.shape[0]), int(k_subset_dist_thresholded.shape[0]))
    s_set_mask_true, no_potentials = make_s_set_mask(
        m_dist_thresholded,
        k_subset_dist_thresholded,
        m_dist_hard,
        k_subset_dist_hard,
        starting_positions,
        max_potential_matches
    )

    # Create S set (candidate matches) from M pixels that passed filtering
    s_set = m_set[s_set_mask_true]
    # Remove K pixels that have no potential matches
    potentials = np.invert(no_potentials)
    k_subset = k_subset[potentials]
    
    log_dataframe_memory(s_set, "S set")
    log_dataframe_memory(k_subset, "K subset (filtered)")

    results = []
    matchless = []

    # Calculate covariance matrix for Mahalanobis distance
    logging.info("Calculating covariance matrix...")
    s_set_for_cov = s_set[DISTANCE_COLUMNS]
    covarience = np.cov(s_set_for_cov, rowvar=False)
    invconv = np.linalg.inv(covarience).astype(np.float32)

    # Free covariance intermediate data
    del s_set_for_cov
    gc.collect()

    # Prepare data arrays for matching algorithm
    s_set_match = s_set[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    s_set_match = np.ascontiguousarray(s_set_match)

    k_subset_match = k_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    k_subset_match = np.ascontiguousarray(k_subset_match)

    logging.info("Starting greedy matching...")
    # Perform greedy matching with random order of K pixels
    add_results, k_idx_matchless = greedy_match_with_shuffled_k(
        k_subset_match,
        s_set_match,
        invconv,
        rng
    )

    # Store match results
    for result in add_results:
        (k_idx, s_idx) = result
        k_row = k_subset.iloc[k_idx]
        match = s_set.iloc[s_idx]

        # Verify hard matches if debug is enabled
        if DEBUG:
            for hard_match_column in hard_match_columns:
                if k_row[hard_match_column] != match[hard_match_column]:
                    raise ValueError("Hard match inconsistency")

        # Collect matched pairs data
        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + DISTANCE_COLUMNS] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + DISTANCE_COLUMNS]
        )

    # Store K pixels that couldn't be matched
    for k_idx in k_idx_matchless:
        k_row = k_subset.iloc[k_idx]
        matchless.append(k_row)

    # Define column names for results DataFrame
    columns = ['k_lat', 'k_lng'] + \
        [f'k_{x}' for x in luc_columns + DISTANCE_COLUMNS] + \
        ['s_lat', 's_lng'] + \
        [f's_{x}' for x in luc_columns + DISTANCE_COLUMNS]

    # Create and save matched pairs DataFrame
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}.parquet'))

    # Create and save unmatched K pixels DataFrame
    matchless_df = pd.DataFrame(matchless, columns=k_subset.columns)
    matchless_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}_matchless.parquet'))

    # Calculate additionality
    logging.info(f"Calculating evaluation year ({evaluation_year}) additionality for {k_grid_id}...")
    additionality_value = 0.0
    if not results_df.empty:
        try:
            # Calculate proportions of each land use class in K pixels
            k_luc_counts = results_df[f"k_luc_{evaluation_year}"].value_counts()
            k_values = np.zeros(len(LandUseClass))
            for luc in LandUseClass:
                if k_luc_counts.get(luc.value) is not None:
                    k_values[luc.value - 1] = k_luc_counts[luc.value]
            k_proportions = k_values / len(results_df)

            # Calculate proportions of each land use class in S pixels
            s_luc_counts = results_df[f"s_luc_{evaluation_year}"].value_counts()
            s_values = np.zeros(len(LandUseClass))
            for luc in LandUseClass:
                if s_luc_counts.get(luc.value) is not None:
                    s_values[luc.value - 1] = s_luc_counts[luc.value]
            s_proportions = s_values / len(results_df)

            # Calculate carbon stocks using land use proportions
            treatment_carbon = (k_proportions * project_area_ha * carbon_density).sum() * MOLECULAR_MASS_CO2_TO_C_RATIO
            control_carbon = (s_proportions * project_area_ha * carbon_density).sum() * MOLECULAR_MASS_CO2_TO_C_RATIO
            
            # Additionality is the difference between treatment and control carbon
            additionality_value = treatment_carbon - control_carbon
            logging.info(f"Iteration {k_grid_id} additionality ({evaluation_year}): {additionality_value:.4f}")
        except KeyError as e:
            logging.error(f"Could not find LUC column for evaluation year {evaluation_year}: {e}")
            additionality_value = np.nan
        except Exception as e:
            logging.error(f"Error calculating additionality for iteration {k_grid_id}: {e}")
            additionality_value = np.nan

    # Clean up large dataframes to free memory
    del m_set, s_set, k_subset, results_df, matchless_df
    del m_dist_thresholded, k_subset_dist_thresholded, m_dist_hard, k_subset_dist_hard
    del s_set_match, k_subset_match, invconv
    gc.collect()
    
    return (k_grid_id, additionality_value)

@jit(nopython=True, fastmath=True, error_model="numpy")
def make_s_set_mask(
    m_dist_thresholded: np.ndarray,
    k_subset_dist_thresholded: np.ndarray,
    m_dist_hard: np.ndarray,
    k_subset_dist_hard: np.ndarray,
    starting_positions: np.ndarray,
    max_potential_matches: int
):
    """Create a boolean mask for M pixels that are potential matches for K pixels."""
    m_size = m_dist_thresholded.shape[0]
    k_size = k_subset_dist_thresholded.shape[0]

    # Boolean masks to track which M pixels to include and which K pixels have no matches
    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    # Process each K pixel
    for k in range(k_size):
        matches = 0
        k_row = k_subset_dist_thresholded[k, :]
        k_hard = k_subset_dist_hard[k]

        # Loop through M pixels, starting from a random position
        for index in range(m_size):
            # Use modulo to wrap around to start of array when reaching the end
            m_index = (index + starting_positions[k]) % m_size

            m_row = m_dist_thresholded[m_index, :]
            m_hard = m_dist_hard[m_index]

            should_include = True

            # Check for exact match on categorical variables
            hard_equals = True
            for j in range(m_hard.shape[0]):
                if m_hard[j] != k_hard[j]:
                    hard_equals = False

            if not hard_equals:
                should_include = False
            else:
                # Check for threshold match on continuous variables
                for j in range(m_row.shape[0]):
                    if abs(m_row[j] - k_row[j]) > 1.0:
                        should_include = False

            # If all criteria are met, include this M pixel
            if should_include:
                s_include[m_index] = True
                matches += 1

            # Stop once we've found enough matches for this K pixel
            if matches == max_potential_matches:
                break

        # Mark K pixels that have no potential matches
        k_miss[k] = matches == 0

    return s_include, k_miss

@jit(nopython=True, fastmath=True, error_model="numpy")
def rows_all_true(rows: np.ndarray):
    """Check if all values in each row are True."""
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
    """Core greedy matching algorithm for finding optimal pairs."""
    # Track which S pixels are still available for matching
    s_available = np.ones((s_set.shape[0],), dtype=np.bool_)
    total_available = s_set.shape[0]

    results = []
    matchless = []

    # Temporary array for storing distances
    s_tmp = np.zeros((s_set.shape[0],), dtype=np.float32)

    # Process K pixels in the specified order
    for k_idx_original in k_order:
        k_row = k_subset[k_idx_original, :]

        # Find S pixels with matching hard criteria that are still available
        hard_matches = rows_all_true(s_set[:, :HARD_COLUMN_COUNT] == k_row[:HARD_COLUMN_COUNT]) & s_available
        hard_matches = hard_matches.reshape(-1)

        if total_available > 0:
            # Calculate Mahalanobis distances for all candidate matches
            s_tmp[hard_matches] = batch_mahalanobis_squared(
                s_set[hard_matches, HARD_COLUMN_COUNT:], k_row[HARD_COLUMN_COUNT:], invcov
            )
            if np.any(hard_matches):
                # Find S pixel with minimum distance
                min_dist_idx = np.argmin(s_tmp[hard_matches])
                s_idx = np.arange(s_tmp.shape[0])[hard_matches][min_dist_idx]

                # Store the match and remove S pixel from available pool
                results.append((k_idx_original, s_idx))
                s_available[s_idx] = False
                total_available -= 1
            else:
                # No hard matches found
                matchless.append(k_idx_original)
        else:
            # No S pixels left
            matchless.append(k_idx_original)

    return (results, matchless)

def greedy_match_with_shuffled_k(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray,
    rng: np.random.Generator
):
    """Wrapper for greedy matching with shuffled K pixel order."""
    k_order = np.arange(k_subset.shape[0])
    rng.shuffle(k_order)
    return greedy_match_core(k_subset, s_set, invcov, k_order)

@jit(nopython=True, fastmath=True, error_model="numpy")
def batch_mahalanobis_squared(rows, vector, invcov):
    """Calculate squared Mahalanobis distances between multiple rows and a single vector."""
    # Calculate difference between each row and the reference vector
    diff = rows - vector
    # Calculate Mahalanobis distance using the inverse covariance matrix
    dists = (np.dot(diff, invcov) * diff).sum(axis=1)
    return dists

def calculate_rse(estimates: list) -> float:
    """Calculate Relative Standard Error (RSE) for a list of estimates."""
    # Filter out NaN values
    valid_estimates = [e for e in estimates if not np.isnan(e)]
    n = len(valid_estimates)
    
    # Need at least 2 values to calculate standard deviation
    if n < 2:
        return np.inf

    mean_val = np.mean(valid_estimates)
    # Avoid division by zero
    if abs(mean_val) < 1e-9:
        return np.inf

    # Calculate standard deviation with n-1 degrees of freedom
    std_dev = np.std(valid_estimates, ddof=1)
    # Standard error = standard deviation / sqrt(sample size)
    std_err = std_dev / np.sqrt(n)
    # Relative standard error = standard error / |mean|
    rse = std_err / abs(mean_val)
    return rse

def calculate_smd(k_values: np.ndarray, s_values: np.ndarray) -> float:
    """Calculate Standardised Mean Difference (SMD) between K and S values."""
    if len(k_values) == 0 or len(s_values) == 0:
        return np.nan
        
    # Calculate means
    mean_k = np.mean(k_values)
    mean_s = np.mean(s_values)
    
    # Calculate variances with n-1 degrees of freedom
    var_k = np.var(k_values, ddof=1) if len(k_values) > 1 else 0
    var_s = np.var(s_values, ddof=1) if len(s_values) > 1 else 0
    
    n_k = len(k_values)
    n_s = len(s_values)
    
    # Calculate pooled variance
    pooled_var = ((n_k - 1) * var_k + (n_s - 1) * var_s) / (n_k + n_s - 2)
    pooled_std = np.sqrt(pooled_var)
    
    # Avoid division by zero
    if pooled_std == 0:
        return 0.0
        
    # SMD = (mean_treatment - mean_control) / pooled_std
    return (mean_k - mean_s) / pooled_std

def analyse_matching_balance(output_folder: str, evaluation_year: int, luc_columns: list, distance_columns: list) -> pd.DataFrame:
    """Analyse balance between treatment and control groups after matching."""
    logging.info("Starting matching balance analysis...")
    
    # Find all matched pair files (excluding matchless files)
    match_files = glob.glob(os.path.join(output_folder, "*.parquet"))
    match_files = [f for f in match_files if not f.endswith("_matchless.parquet")]
    if not match_files:
        logging.warning("No matched parquet files found for balance analysis")
        return pd.DataFrame()
        
    logging.info(f"Found {len(match_files)} matched parquet files")
    
    # Initialize dictionaries to collect all K and S values
    all_k_data = {var: [] for var in distance_columns}
    all_s_data = {var: [] for var in distance_columns}
    
    # Process each matched pair file
    for match_file in match_files:
        try:
            df = pd.read_parquet(match_file)
            if len(df) == 0:
                continue
                
            # Collect values for each continuous variable
            for var in distance_columns:
                k_col = f'k_{var}'
                s_col = f's_{var}'
                if k_col in df.columns and s_col in df.columns:
                    all_k_data[var].extend(df[k_col].dropna().values)
                    all_s_data[var].extend(df[s_col].dropna().values)
        except Exception as e:
            logging.warning(f"Error processing {match_file}: {e}")
            continue
            
    # Calculate SMD for each variable
    smd_results = []
    for var in distance_columns:
        k_vals = np.array(all_k_data[var])
        s_vals = np.array(all_s_data[var])
        
        if len(k_vals) > 0 and len(s_vals) > 0:
            smd = calculate_smd(k_vals, s_vals)
            smd_results.append({
                'variable': var,
                'n_k': len(k_vals),
                'n_s': len(s_vals),
                'mean_k': np.mean(k_vals),
                'mean_s': np.mean(s_vals),
                'std_k': np.std(k_vals, ddof=1),
                'std_s': np.std(s_vals, ddof=1),
                'smd': smd,
                'abs_smd': abs(smd)
            })
        else:
            smd_results.append({
                'variable': var,
                'n_k': len(k_vals),
                'n_s': len(s_vals),
                'mean_k': np.nan,
                'mean_s': np.nan,
                'std_k': np.nan,
                'std_s': np.nan,
                'smd': np.nan,
                'abs_smd': np.nan
            })
            
    smd_df = pd.DataFrame(smd_results)
    
    # Add interpretation of SMD values
    def interpret_smd(smd_val):
        if np.isnan(smd_val):
            return "No data"
        abs_smd = abs(smd_val)
        if abs_smd < 0.1:
            return "Negligible"
        elif abs_smd < 0.2:
            return "Small"
        elif abs_smd < 0.5:
            return "Medium"
        elif abs_smd < 0.8:
            return "Large"
        else:
            return "Very large"
            
    smd_df['smd_interpretation'] = smd_df['smd'].apply(interpret_smd)
    logging.info("Completed matching balance analysis")
    return smd_df

def extract_k_grid_number(filepath: str) -> int:
    """Extract the numeric part from K grid filenames for proper sorting."""
    basename = os.path.basename(filepath)
    match = re.search(r'k_(\d+)\.parquet$', basename)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"Could not extract grid number from {basename}. Placing it at the end.")
        return float('inf')

def create_single_m_sample(args):
    """Create a single M sample file - designed for multiprocessing."""
    m_set, sample_index, sample_size, seed, m_samples_dir = args
    
    logging.info(f"Creating sample {sample_index+1} with seed {seed}")
    
    # Sample the M set
    should_replace = len(m_set) < sample_size
    if should_replace:
        sample = m_set.sample(n=sample_size, random_state=seed, replace=True).reset_index(drop=True)
    else:
        sample = m_set.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)
    
    # Shuffle the sampled rows to remove any residual ordering
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    
    # Save sample
    sample_filename = os.path.join(m_samples_dir, f"m_sample_{sample_index+1:03d}.parquet")
    sample.to_parquet(sample_filename)
    
    # Clean up sample from memory
    del sample
    gc.collect()
    
    return f"Completed sample {sample_index+1}"

def create_m_samples(
    m_parquet_filename: str,
    output_folder: str,
    num_samples: int,
    sample_size: int,
    seeds: np.ndarray,
    sample_processes: int = None
) -> None:
    """Create pre-sampled M set files for each matching iteration using multiprocessing."""
    if sample_processes is None:
        sample_processes = min(cpu_count(), 8)
    
    # Create m_samples directory
    m_samples_dir = os.path.join(output_folder, "m_samples")
    os.makedirs(m_samples_dir, exist_ok=True)
    
    # Check which samples already exist
    existing_samples = []
    missing_samples = []
    
    for i in range(num_samples):
        sample_filename = os.path.join(m_samples_dir, f"m_sample_{i+1:03d}.parquet")
        if os.path.exists(sample_filename):
            existing_samples.append(i+1)
        else:
            missing_samples.append(i)
    
    if existing_samples:
        logging.info(f"Found {len(existing_samples)} existing M samples")
    
    if not missing_samples:
        logging.info("All M samples already exist. Skipping sample creation.")
        return
    
    logging.info(f"Creating {len(missing_samples)} missing M set samples of {sample_size:,} rows each using {sample_processes} processes...")
    
    # Load full M set once
    logging.info(f"Loading full M set from {m_parquet_filename}")
    m_set = pd.read_parquet(m_parquet_filename)
    
    log_dataframe_memory(m_set, "M set (full)")
    
    # Prepare arguments for multiprocessing
    sample_args = []
    for i in missing_samples:
        sample_args.append((m_set, i, sample_size, seeds[i], m_samples_dir))
    
    # Create samples in parallel
    with Pool(processes=sample_processes) as pool:
        results = pool.map(create_single_m_sample, sample_args)
    
    # Log results
    for result in results:
        logging.info(result)
    
    # Clean up full M set
    del m_set
    gc.collect()
    
    logging.info(f"Completed creating {len(missing_samples)} missing M set samples in {m_samples_dir}")

def iteration_func_wrapper(args):
    """Wrapper function for multiprocessing - must be at module level to be picklable."""
    m_sample_file, k_grid_file, iter_seed, shuf_seed, start_year, evaluation_year, carbon_density, project_area_ha, output_folder = args
    return find_match_iteration(
        m_sample_file,
        start_year,
        evaluation_year,
        carbon_density,
        project_area_ha,
        output_folder,
        (k_grid_file, iter_seed),
        shuf_seed
    )

def find_pairs(
    k_directory: str,
    m_parquet_filename: str,    # now points directly at your full matches.parquet
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
    """Main matching function that processes K grids in batches until convergence."""
    logging.info("Starting find pairs (no sampling / no shuffle)")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup memory logging
    setup_memory_logging(output_folder)
    log_memory_usage("main function start", detailed=True)
    log_system_memory()
    
    # Check if M set file exists
    if not os.path.exists(m_parquet_filename):
        logging.error(f"M set file not found: {m_parquet_filename}")
        return
    
    # Find all K grid files
    k_grid_files_unsorted = glob.glob(os.path.join(k_directory, "k_*.parquet"))
    if not k_grid_files_unsorted:
        logging.error(f"No k_*.parquet files found in directory: {k_directory}")
        return
    
    k_grid_files = sorted(k_grid_files_unsorted, key=extract_k_grid_number)
    num_k_grids = len(k_grid_files)
    logging.info(f"Found and numerically sorted {num_k_grids} K grid files to process.")
    
    # Generate all random seeds upfront
    rng = np.random.default_rng(seed)
    iteration_seeds = rng.integers(0, 2000000, num_k_grids)
    shuffle_seeds = rng.integers(0, 2000000, num_k_grids)
    
    # Prepare pool arguments: every iteration uses the same M file
    map_arguments = []
    for i, (k_grid_file, iter_seed, shuf_seed) in enumerate(
            zip(k_grid_files, iteration_seeds, shuffle_seeds)
    ):
        m_sample_file = m_parquet_filename
        map_arguments.append((
            m_sample_file,
            k_grid_file,
            iter_seed,
            shuf_seed,         # still passed but we'll ignore shuffle below
            start_year,
            evaluation_year,
            carbon_density,
            project_area_ha,
            output_folder
        ))
    
    all_additionality_estimates = []
    processed_k_grid_ids = []
    total_processed = 0
    
    log_memory_usage("before starting multiprocessing pool", detailed=True)
    
    with Pool(processes=processes_count) as pool:
        for i in range(0, num_k_grids, batch_size):
            batch_args = map_arguments[i:min(i + batch_size, num_k_grids)]
            if not batch_args:
                break
            
            logging.info(f"Processing batch {i//batch_size + 1}/{(num_k_grids + batch_size - 1)//batch_size}")
            
            log_memory_usage(f"before batch {i//batch_size + 1}", detailed=True)
            log_system_memory()
            
            # Use the module-level function directly
            batch_results = pool.map(iteration_func_wrapper, batch_args)
            
            log_memory_usage(f"after batch {i//batch_size + 1}", detailed=True)
            
            for k_grid_id, additionality_value in batch_results:
                processed_k_grid_ids.append(k_grid_id)
                all_additionality_estimates.append(additionality_value)
            
            total_processed = len(all_additionality_estimates)
            current_rse = calculate_rse(all_additionality_estimates)
            
            logging.info(f"Processed {total_processed}/{num_k_grids} K grids. Current RSE: {current_rse:.4f} (Threshold: {rse_threshold:.4f})")
            
            if current_rse <= rse_threshold:
                logging.info(f"Convergence reached (RSE <= {rse_threshold}). Stopping.")
                break
    
    log_memory_usage("after multiprocessing completed", detailed=True)
    logging.info(f"Finished processing. Total K grids processed: {total_processed}")
    
    # Calculate and save matching balance metrics
    logging.info("Calculating Standardised Mean Differences for matching balance...")
    log_memory_usage("before SMD analysis")
    
    # Get luc_columns from first matched file
    first_match = glob.glob(os.path.join(output_folder, "[0-9]*.parquet"))
    if first_match:
        luc_columns = [x for x in pd.read_parquet(first_match[0]).columns if x.startswith('luc')]
    else:
        luc_columns = []
    
    distance_columns = DISTANCE_COLUMNS
    smd_df = analyse_matching_balance(output_folder, evaluation_year, luc_columns, distance_columns)
    
    if not smd_df.empty:
        smd_output_file = os.path.join(output_folder, "matching_balance_smd.csv")
        smd_df.to_csv(smd_output_file, index=False)
        logging.info(f"SMD analysis saved to: {smd_output_file}")
        
        # Print SMD results table
        print("\n" + "="*80)
        print("MATCHING BALANCE ANALYSIS - Standardised Mean Differences (SMD)")
        print("="*80)
        print(f"{'Variable':<15} {'N_K':<8} {'N_S':<8} {'Mean_K':<10} {'Mean_S':<10} {'SMD':<8} {'Interpretation'}")
        print("-"*80)
        for _, row in smd_df.iterrows():
            print(f"{row['variable']:<15} {row['n_k']:<8} {row['n_s']:<8} "
                  f"{row['mean_k']:<10.3f} {row['mean_s']:<10.3f} "
                  f"{row['smd']:<8.3f} {row['smd_interpretation']}")
        print("-"*80)
        print(f"SMD Interpretation: |SMD| < 0.1 = Negligible, 0.1-0.2 = Small, 0.2-0.5 = Medium, 0.5-0.8 = Large, >0.8 = Very Large")
        print(f"Good matching typically has |SMD| < 0.1 for all variables")
        print("="*80)
    else:
        logging.warning("Could not calculate SMD analysis - no valid matched data found")
    
    log_memory_usage("end of main function", detailed=True)
    
    # Write final summary to memory log
    if MEMORY_LOG_FILE:
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL SUMMARY\n")
            f.write(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total K grids processed: {total_processed}\n")
            f.write(f"Processes used: {processes_count}\n")
            f.write(f"{'='*80}\n")

def main():
    """Main entry point for the matching algorithm."""
    parser = argparse.ArgumentParser(description="Match K→M without sampling or multiprocessing")
    parser.add_argument("--k_directory", required=True)
    parser.add_argument("--m_parquet_filename", required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--evaluation_year", type=int, required=True)
    parser.add_argument("--carbon_density", required=True)
    parser.add_argument("--project_area", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--rse_threshold", type=float, default=0.05,
                       help="Stop early if cumulative RSE ≤ this value")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Number of K grids to process per batch")
    parser.add_argument("--processes_count", type=int, default=16,
                       help="Number of processes to use for parallel processing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Load carbon density
    logging.info("Loading carbon density")
    carbon_density_dict = load_carbon_density(args.carbon_density)
    carbon_density = np.zeros(6)  # Assuming 6 land use classes
    for luc, density in carbon_density_dict.items():
        if 1 <= luc <= 6:
            carbon_density[luc-1] = density
    
    # Compute project area
    logging.info("Computing project area")
    proj_area_ha = compute_project_area_ha(args.project_area)
    logging.info(f"Project area: {proj_area_ha:.2f} ha")
    
    # Run the main matching process
    find_pairs(
        k_directory=args.k_directory,
        m_parquet_filename=args.m_parquet_filename,
        start_year=args.start_year,
        evaluation_year=args.evaluation_year,
        carbon_density=carbon_density,
        project_area_ha=proj_area_ha,
        seed=args.seed,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        rse_threshold=args.rse_threshold,
        processes_count=args.processes_count
    )

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()