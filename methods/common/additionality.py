import glob
import os
import logging
from typing import Dict, Any, List, cast

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from geojson import LineString, FeatureCollection, Feature, MultiPoint, dumps  # type: ignore

from methods.common import LandUseClass

MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def plot_carbon_stock(
    axis: plt.Axes,
    project_data: Dict[int, float],
    control_data: Dict[int, float],
    start_year: int
) -> None:
    """Will plot the carbon stock for a project and the controls. Those dictionaries should
    be the yearly carbon stock."""
    x_axis = []
    treatment = []
    control = []
    for year, value in project_data.items():
        x_axis.append(year)
        treatment.append(value)
        control.append(control_data[year])
    axis.plot(x_axis, treatment, label="Treatment")
    axis.plot(x_axis, control, label="Control")
    axis.set_title("Carbon stock (Average Treatment and Average Control)")
    axis.set_xlabel("Year")
    axis.set_ylabel("Carbon Stock (MgCO2e)")
    axis.axvline(start_year)
    axis.legend(loc="lower left")


def plot_carbon_trajectories(
    axis: List[plt.Axes],
    title: str,
    idx: int,
    timeseries: Dict[int, np.ndarray],
    start_year: str
):
    x_axis = []
    y_axis = []
    for year, value in timeseries.items():
        x_axis.append(year)
        y_axis.append(value)
    axis[idx].plot(x_axis, y_axis)
    axis[idx].set_title(title)
    axis[idx].set_xlabel("Year")
    axis[idx].set_ylabel("Carbon Stock (MgCO2e)")
    axis[idx].axvline(int(start_year))


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
    matches_directory: str,
    partials_dir: str | None = None # Optional: For saving plots/diagnostics
) -> pd.DataFrame: # Return a DataFrame
    """
    Calculate additionality, carbon stocks, avoided deforestation, and associated
    standard errors (SE) and relative standard errors (RSE) for a project
    based on counterfactual pair matchings.

    Args:
        project_area_msq: Area of the project in square meters.
        project_start: The start year of the project (as string or int).
        end_year: The final year for analysis.
        density: Numpy array of carbon densities per land use class.
        matches_directory: Directory containing the pairs parquet files (output of find_pairs).
        partials_dir: Optional directory to save diagnostic plots and files.

    Returns:
        A pandas DataFrame with yearly metrics including means, SE, and RSE.
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

    try:
        deforested_luc_index = LandUseClass.DEFORESTED.value - 1
    except AttributeError:
        logging.error("LandUseClass.DEFORESTED not found. Cannot calculate deforestation metrics.")
        deforested_luc_index = -1

    for year in all_years:
        treatment_props_year = treatment_luc_proportions.get(year, np.full((num_iterations, len(LandUseClass)), np.nan))
        control_props_year = control_luc_proportions.get(year, np.full((num_iterations, len(LandUseClass)), np.nan))

        treatment_carbon_iter = np.nansum(treatment_props_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
        control_carbon_iter = np.nansum(control_props_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
        additionality_iter = treatment_carbon_iter - control_carbon_iter

        valid_indices = ~np.isnan(additionality_iter)
        valid_additionality = additionality_iter[valid_indices]
        valid_treatment_carbon = treatment_carbon_iter[valid_indices]
        valid_control_carbon = control_carbon_iter[valid_indices]
        n_valid = len(valid_additionality)

        mean_additionality = np.mean(valid_additionality) if n_valid > 0 else np.nan

        if n_valid >= 2:
            std_dev_additionality = np.std(valid_additionality, ddof=1)
            stderr_additionality = std_dev_additionality / np.sqrt(n_valid)
            if abs(mean_additionality) > 1e-9:
                rse_additionality = stderr_additionality / abs(mean_additionality)
            else:
                rse_additionality = np.inf
        else:
            stderr_additionality = np.nan
            rse_additionality = np.nan

        mean_treatment_carbon = np.mean(valid_treatment_carbon) if n_valid > 0 else np.nan
        mean_control_carbon = np.mean(valid_control_carbon) if n_valid > 0 else np.nan
        stderr_treatment_carbon = (np.std(valid_treatment_carbon, ddof=1) / np.sqrt(n_valid)) if n_valid >= 2 else np.nan
        stderr_control_carbon = (np.std(valid_control_carbon, ddof=1) / np.sqrt(n_valid)) if n_valid >= 2 else np.nan

        mean_avoided_deforestation = np.nan
        stderr_avoided_deforestation = np.nan
        if deforested_luc_index != -1:
             treatment_deforested_area_iter = treatment_props_year[:, deforested_luc_index] * project_area_ha
             control_deforested_area_iter = control_props_year[:, deforested_luc_index] * project_area_ha
             avoided_deforestation_iter = control_deforested_area_iter - treatment_deforested_area_iter

             valid_avoided_deforestation = avoided_deforestation_iter[valid_indices]
             valid_treatment_deforested = treatment_deforested_area_iter[valid_indices]
             valid_control_deforested = control_deforested_area_iter[valid_indices]

             mean_avoided_deforestation = np.mean(valid_avoided_deforestation) if n_valid > 0 else np.nan
             if n_valid >= 2:
                 stderr_avoided_deforestation = np.std(valid_avoided_deforestation, ddof=1) / np.sqrt(n_valid)

        results_list.append({
            "year": year,
            "iterations_valid": n_valid,
            "additionality_mean": mean_additionality,
            "additionality_stderr": stderr_additionality,
            "additionality_rse": rse_additionality,
        })

    results_df = pd.DataFrame(results_list)

    if partials_dir is not None and num_iterations > 0 and not results_df.empty:
        try:
            logging.info(f"Generating diagnostic plots and files in {partials_dir}")
            os.makedirs(partials_dir, exist_ok=True)

            plot_year = all_years[-1]
            plot_data = results_df[results_df['year'] == plot_year].iloc[0]
            treatment_props_plot_year = treatment_luc_proportions.get(plot_year)
            control_props_plot_year = control_luc_proportions.get(plot_year)

            if treatment_props_plot_year is not None and control_props_plot_year is not None:
                treatment_carbon_all_iters = np.nansum(treatment_props_plot_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
                control_carbon_all_iters = np.nansum(control_props_plot_year * project_area_ha * density, axis=1) * MOLECULAR_MASS_CO2_TO_C_RATIO
                valid_indices_plot = ~np.isnan(treatment_carbon_all_iters) & ~np.isnan(control_carbon_all_iters)

                figure, untyped_axis = plt.subplots(1, 3, figsize=(18, 6))
                axis = cast(List[plt.Axes], untyped_axis)

                p_tot_plot = results_df.set_index('year')['treatment_carbon_mean'].dropna().to_dict()
                c_tot_plot = results_df.set_index('year')['control_carbon_mean'].dropna().to_dict()
                if p_tot_plot and c_tot_plot:
                     plot_carbon_stock(axis[0], p_tot_plot, c_tot_plot, int(project_start))
                else:
                     axis[0].set_title("Carbon stock (Average) - No Data")

                if np.any(valid_indices_plot):
                    axis[1].hist(treatment_carbon_all_iters[valid_indices_plot], bins=20, alpha=0.7)
                    axis[1].set_title(f'Treatment Carbon Distribution ({plot_year})')
                    axis[1].set_xlabel('Carbon Stock (MgCO2e)')
                    axis[1].set_ylabel('Frequency')

                    axis[2].hist(control_carbon_all_iters[valid_indices_plot], bins=20, alpha=0.7)
                    axis[2].set_title(f'Control Carbon Distribution ({plot_year})')
                    axis[2].set_xlabel('Carbon Stock (MgCO2e)')
                    axis[2].set_ylabel('Frequency')
                else:
                     axis[1].set_title(f'Treatment Carbon ({plot_year}) - No Data')
                     axis[2].set_title(f'Control Carbon ({plot_year}) - No Data')

                plt.tight_layout()
                out_path_plot = os.path.join(partials_dir, "summary_carbon_stock.png")
                figure.savefig(out_path_plot)
                plt.close(figure)
                logging.info(f"Saved summary plot to {out_path_plot}")

            first_valid_match_file = None
            first_valid_match_df = None
            for pairs_file in matches:
                 try:
                     df_temp = pd.read_parquet(os.path.join(matches_directory, pairs_file))
                     if not df_temp.empty:
                         first_valid_match_file = pairs_file
                         first_valid_match_df = df_temp
                         break
                 except Exception:
                     continue

            if first_valid_match_file and first_valid_match_df is not None:
                logging.info(f"Generating diagnostics (SMD, GeoJSON) using: {first_valid_match_file}")
                smds : Dict[str, Any] = {"pair_id": [], "feature": [], "smd": []}
                mean_std = first_valid_match_df.agg(["mean", "std"])
                for col in first_valid_match_df.columns:
                    if col.startswith("k_"):
                        feature = "_".join(col.split("_")[1:])
                        control_col = "s_" + feature
                        if control_col in mean_std and col in mean_std:
                            treat_mean = mean_std[col]["mean"]
                            control_mean = mean_std[control_col]["mean"]
                            treat_std = mean_std[col]["std"]
                            control_std = mean_std[control_col]["std"]
                            denominator = np.sqrt((treat_std**2 + control_std**2) / 2)
                            if denominator > 1e-9:
                                smd = abs(treat_mean - control_mean) / denominator
                            else:
                                smd = 0.0 if abs(treat_mean - control_mean) < 1e-9 else np.inf
                            smds["pair_id"].append(os.path.splitext(first_valid_match_file)[0])
                            smds["feature"].append(feature)
                            smds["smd"].append(round(smd, 8))

                if smds["pair_id"]:
                    smd_path = os.path.join(partials_dir, "smd_summary.csv")
                    smds_df = pd.DataFrame.from_dict(smds)
                    smds_df.to_csv(smd_path, index=False)
                    logging.info(f"Saved SMD summary to {smd_path}")

                linestrings = []
                points = []
                for _, row in first_valid_match_df.iterrows():
                     if all(pd.notna(coord) for coord in [row.get("k_lng"), row.get("k_lat"), row.get("s_lng"), row.get("s_lat")]):
                         try:
                             linestring = Feature(geometry=LineString([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                             point = Feature(geometry=MultiPoint([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                             linestrings.append(linestring)
                             points.append(point)
                         except Exception as geo_e:
                             logging.warning(f"Could not create geometry for row in {first_valid_match_file}: {geo_e}")

                if linestrings:
                     geom_collection_lines = FeatureCollection(linestrings)
                     out_path_lines = os.path.join(partials_dir, f"{os.path.splitext(first_valid_match_file)[0]}-pairs.geojson")
                     with open(out_path_lines, "w", encoding="utf-8") as f: f.write(dumps(geom_collection_lines))
                     logging.info(f"Saved pairs linestrings to {out_path_lines}")
                if points:
                     geom_collection_points = FeatureCollection(points)
                     out_path_points = os.path.join(partials_dir, f"{os.path.splitext(first_valid_match_file)[0]}-pairs-points.geojson")
                     with open(out_path_points, "w", encoding="utf-8") as f: f.write(dumps(geom_collection_points))
                     logging.info(f"Saved pairs points to {out_path_points}")

            else:
                 logging.warning("No valid iteration data found for generating SMD/GeoJSON diagnostics.")

        except Exception as e:
             logging.error(f"Failed during diagnostic generation: {e}", exc_info=True)

    return results_df
