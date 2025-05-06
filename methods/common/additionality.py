import glob
import os
import logging
from typing import Dict, Any, List, cast

import numpy as np # type: ignore
import pandas as pd # type: ignore
from methods.common import LandUseClass

MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def find_first_luc(columns: list[str]) -> int:
    """Finds the earliest year present in LUC column names (e.g., k_luc_2010)."""
    min_year = float('inf')
    found = False
    for col in columns:
        if "_luc_" in col:
            try:
                year = int(col.split("_luc_")[-1])
                min_year = min(min_year, year)
                found = True
            except (ValueError, IndexError):
                continue
    if not found:
        raise ValueError("Failed to extract any year from LUC columns")
    return int(min_year)


def is_not_matchless(path: str) -> bool:
    """Checks if a filename does not end with _matchless.parquet."""
    return not path.endswith("_matchless.parquet")


def generate_additionality(
    project_area_msq: float,
    project_start: str,
    end_year: int,
    density: np.ndarray,
    matches_directory: str
) -> pd.DataFrame: # Return a DataFrame
    """
    Calculate additionality mean, standard error (SE), and relative standard
    error (RSE) for a project based on counterfactual pair matchings.

    Args:
        project_area_msq: Area of the project in square meters.
        project_start: The start year of the project (as string or int). Not directly used.
        end_year: The final year for analysis.
        density: Numpy array of carbon densities per land use class.
        matches_directory: Directory containing the pairs parquet files (output of find_pairs).
        # Removed partials_dir from Args

    Returns:
        A pandas DataFrame with yearly additionality metrics (mean, SE, RSE).
    """
    project_area_ha = project_area_msq / 10000.0 # Convert m^2 to hectares
    logging.info(f"Project area: {project_area_msq:.2f} m^2 ({project_area_ha:.2f} ha)")

    # Find all non-matchless pairs files
    matches = glob.glob("*.parquet", root_dir=matches_directory)
    matches = [x for x in matches if is_not_matchless(x)]
    num_iterations = len(matches)

    if num_iterations == 0:
        raise ValueError(f"No non-matchless parquet files found in {matches_directory}")
    logging.info(f"Found {num_iterations} match files (iterations) to process.")

    # Dictionaries to store LUC proportions per iteration for each year
    treatment_luc_proportions: Dict[int, np.ndarray] = {}
    control_luc_proportions: Dict[int, np.ndarray] = {}
    earliest_year_overall = float('inf')

    # --- Loop 1: Extract LUC proportions from each iteration file ---
    for pair_idx, pairs_file in enumerate(matches):
        logging.debug(f"Processing iteration {pair_idx + 1}/{num_iterations}: {pairs_file}")
        file_path = os.path.join(matches_directory, pairs_file)
        try:
            matches_df = pd.read_parquet(file_path)
        except Exception as e:
            logging.error(f"Failed to read parquet file {file_path}: {e}")
            continue # Skip this file

        if matches_df.empty:
            logging.warning(f"Skipping empty pairs file: {pairs_file}")
            continue

        columns = matches_df.columns.to_list()
        try:
            earliest_year_in_file = find_first_luc(columns)
            earliest_year_overall = min(earliest_year_overall, earliest_year_in_file)
        except ValueError as e:
            logging.error(f"Could not determine start year for {pairs_file}: {e}")
            continue # Skip this file if years can't be determined

        # Process each year present in the file, up to the overall end_year
        for year_index in range(earliest_year_in_file, end_year + 1):
            k_luc_col = f"k_luc_{year_index}"
            s_luc_col = f"s_luc_{year_index}"

            # Check if columns for the year exist
            if k_luc_col not in columns or s_luc_col not in columns:
                logging.warning(f"LUC columns for year {year_index} not found in {pairs_file}. Skipping year.")
                continue

            # --- Treatment Proportions ---
            total_pixels_t = len(matches_df)
            values_t = np.zeros(len(LandUseClass))
            value_count_year_t = matches_df[k_luc_col].value_counts()
            for luc in LandUseClass:
                if value_count_year_t.get(luc.value) is not None:
                    if 0 <= luc.value - 1 < len(values_t):
                         values_t[luc.value - 1] = value_count_year_t[luc.value]
                    else:
                        logging.warning(f"Invalid LUC value {luc.value} encountered in {pairs_file}, year {year_index}.")

            proportions_t = values_t / total_pixels_t
            prop_t_sum = np.sum(proportions_t)
            if not (0.99 < prop_t_sum < 1.01):
                 logging.warning(f"Treatment proportions sum to {prop_t_sum:.4f} for {pairs_file}, year {year_index}")

            # Initialize array for the year if first time seeing it
            if treatment_luc_proportions.get(year_index) is None:
                treatment_luc_proportions[year_index] = np.full((num_iterations, len(LandUseClass)), np.nan)
            treatment_luc_proportions[year_index][pair_idx, :] = proportions_t

            # --- Control Proportions ---
            total_pixels_c = len(matches_df)
            values_c = np.zeros(len(LandUseClass))
            value_count_year_c = matches_df[s_luc_col].value_counts()
            for luc in LandUseClass:
                 if value_count_year_c.get(luc.value) is not None:
                    if 0 <= luc.value - 1 < len(values_c):
                        values_c[luc.value - 1] = value_count_year_c[luc.value]
                    else:
                        logging.warning(f"Invalid LUC value {luc.value} encountered in {pairs_file}, year {year_index}.")

            proportions_c = values_c / total_pixels_c
            prop_c_sum = np.sum(proportions_c)
            if not (0.99 < prop_c_sum < 1.01):
                 logging.warning(f"Control proportions sum to {prop_c_sum:.4f} for {pairs_file}, year {year_index}")

            if control_luc_proportions.get(year_index) is None:
                control_luc_proportions[year_index] = np.full((num_iterations, len(LandUseClass)), np.nan)
            control_luc_proportions[year_index][pair_idx, :] = proportions_c

    if earliest_year_overall == float('inf'):
        raise ValueError("Could not determine earliest year from any input file.")

    # --- Loop 2: Calculate final metrics for each year ---
    results_list = []
    all_years = sorted([y for y in treatment_luc_proportions.keys() if y >= earliest_year_overall])

    for year in all_years:
        treatment_props_year = treatment_luc_proportions.get(year, np.full((num_iterations, len(LandUseClass)), np.nan))
        control_props_year = control_luc_proportions.get(year, np.full((num_iterations, len(LandUseClass)), np.nan))

        # --- Carbon Calculations ---
        treatment_carbon_iter = np.nansum(treatment_props_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
        control_carbon_iter = np.nansum(control_props_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
        additionality_iter = treatment_carbon_iter - control_carbon_iter

        # Filter out NaN results before calculating stats
        valid_indices = ~np.isnan(additionality_iter)
        valid_additionality = additionality_iter[valid_indices]
        n_valid = len(valid_additionality) # Number of valid iterations for this year

        # Calculate Mean Additionality
        mean_additionality = np.mean(valid_additionality) if n_valid > 0 else np.nan

        # Calculate Standard Error (SE) and Relative Standard Error (RSE) for Additionality
        if n_valid >= 2:
            std_dev_additionality = np.std(valid_additionality, ddof=1)
            stderr_additionality = std_dev_additionality / np.sqrt(n_valid)
            # Calculate RSE, handle mean close to zero
            if abs(mean_additionality) > 1e-9:
                rse_additionality = stderr_additionality / abs(mean_additionality)
            else:
                rse_additionality = np.inf # Assign infinity if mean is effectively zero
        else:
            stderr_additionality = np.nan
            rse_additionality = np.nan

        # Append results for the year
        results_list.append({
            "year": year,
            "iterations_valid": n_valid,
            "additionality_mean": mean_additionality,
            "additionality_stderr": stderr_additionality,
            "additionality_rse": rse_additionality,
        })

    # Convert list of results to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df