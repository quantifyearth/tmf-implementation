import glob
import os
import logging
import csv
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional
from geojson import LineString, FeatureCollection, Feature, MultiPoint, dumps # type: ignore

from methods.common import LandUseClass, dump_dir
from methods.common.geometry import area_for_geometry

EXPECTED_NUMBER_OF_MATCH_ITERATIONS = 100
MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def plot_carbon_stock(axis, arr, cf, start_year):
    x_axis = []
    treatment = []
    control = []
    for (k, v) in arr.items():
        x_axis.append(k)
        treatment.append(v)
        control.append(cf[k])
    axis[0].plot(x_axis, treatment, label="Treatment")
    axis[0].plot(x_axis, control, label="Control")
    axis[0].set_title('Carbon stock (Average Treatment and Average Control)')
    axis[0].set_xlabel('Year')
    axis[0].set_ylabel('Carbon Stock (MgCO2e)')
    axis[0].axvline(start_year)
    axis[0].legend(loc="lower left")

def plot_carbon_trajectories(axis, title, idx, ts, start_year):
    x_axis = []
    y_axis = []
    for (k, v) in ts.items():
        x_axis.append(k)
        y_axis.append(ts[k])
    axis[idx].plot(x_axis, y_axis)
    axis[idx].set_title(title)
    axis[idx].set_xlabel('Year')
    axis[idx].set_ylabel('Carbon Stock (MgCO2e)')
    axis[idx].axvline(start_year)

def find_first_luc(columns: list[str]) -> Optional[str]:
    for col in columns:
        split = col.split("_luc_")
        if len(split) < 2:
            continue
        try:
            return int(split[1])
        except:
            continue
    return None

def is_not_matchless(path: str) -> bool:
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 2:
        return True
    else:
        if parts[1] == "matchless.parquet":
            return False
    return True

def generate_additionality(
    project_geojson_file: str,
    project_start: str,
    end_year: int,
    carbon_density: str,
    matches_directory: str,
    output_csv: str,
) -> int:
    # TODO: may be present in config, in which case use that, but for now we use
    # the calculate version.
    density_df = pd.read_csv(carbon_density)
    density = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Density may have left some LUCs out like water
    for _, row in density_df.iterrows():
        luc = row["land use class"]
        density[int(luc) - 1] = row["carbon density"]

    # We calculate area using projections and not the inaccurate 30 * 30 approximation
    project_gpd = gpd.read_file(project_geojson_file)
    project_area_msq = area_for_geometry(project_gpd)

    logging.info("Project area: %.2fmsq", project_area_msq)

    matches = glob.glob("*.parquet", root_dir=matches_directory)
    matches = list(filter(is_not_matchless, matches))
    assert len(matches) == EXPECTED_NUMBER_OF_MATCH_ITERATIONS

    treatment_data = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing additionality in treatment for %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        columns = matches_df.columns.to_list()
        columns.sort()

        earliest_year = find_first_luc(columns)

        if earliest_year is None:
            raise ValueError("Failed to extract earliest year from LUCs")

        for year_index in range(earliest_year, end_year + 1):
            total_pixels_t = len(matches_df)

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"k_luc_{year_index}"].value_counts()

            for luc in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                if value_count_year.get(luc) is not None:
                    values[int(luc) - 1] = value_count_year[luc]

            undisturbed_t = values[LandUseClass.UNDISTURBED - 1]
            degraded_t = values[LandUseClass.DEGRADED - 1]
            deforested_t = values[LandUseClass.DEFORESTED - 1]
            regrowth_t = values[LandUseClass.REGROWTH - 1]
            water_t = values[LandUseClass.WATER - 1]
            other_t = values[LandUseClass.OTHER - 1]

            proportions_t = np.array(
                [
                    undisturbed_t / total_pixels_t,
                    degraded_t / total_pixels_t,
                    deforested_t / total_pixels_t,
                    regrowth_t / total_pixels_t,
                    water_t / total_pixels_t,
                    other_t / total_pixels_t,
                ]
            )

            # Quick Sanity Check
            prop = np.sum(proportions_t)
            assert 0.99 < prop < 1.01

            areas_t = proportions_t * (project_area_msq / 10000)
            s_t = areas_t * density

            s_t_value = s_t.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            logging.info("Additionality in treatment is %f", s_t_value)

            if treatment_data.get(year_index) is not None:
                treatment_data[year_index][pair_idx] = s_t_value
            else:
                arr = [0 for _ in range(EXPECTED_NUMBER_OF_MATCH_ITERATIONS)]
                arr[pair_idx] = s_t_value
                treatment_data[year_index] = arr

    scvt = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing additionality for control %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        columns = matches_df.columns.to_list()
        columns.sort()

        earliest_year = find_first_luc(columns)

        if earliest_year is None:
            raise ValueError("Failed to extract earliest year from LUCs")

        total_pixels_c = len(matches_df)
        for year_index in range(earliest_year, end_year + 1):

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"s_luc_{year_index}"].value_counts()

            for luc in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                if value_count_year.get(luc) is not None:
                    values[int(luc) - 1] = value_count_year[luc]

            undisturbed_c = values[LandUseClass.UNDISTURBED - 1]
            degraded_c = values[LandUseClass.DEGRADED - 1]
            deforested_c = values[LandUseClass.DEFORESTED - 1]
            regrowth_c = values[LandUseClass.REGROWTH - 1]
            water_c = values[LandUseClass.WATER - 1]
            other_c = values[LandUseClass.OTHER - 1]

            proportions_c = np.array(
                [
                    undisturbed_c / total_pixels_c,
                    degraded_c / total_pixels_c,
                    deforested_c / total_pixels_c,
                    regrowth_c / total_pixels_c,
                    water_c / total_pixels_c,
                    other_c / total_pixels_c,
                ]
            )

            # Quick Sanity Check
            prop = np.sum(proportions_c)
            assert 0.99 < prop < 1.01

            areas_c = proportions_c * (project_area_msq / 10000)
            s_c = areas_c * density

            s_c_value = s_c.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            logging.info("Additionality in counterfactual is %f", s_c_value)

            if scvt.get(year_index) is not None:
                scvt[year_index][pair_idx] = s_c_value
            else:
                arr = [0 for _ in range(EXPECTED_NUMBER_OF_MATCH_ITERATIONS)]
                arr[pair_idx] = s_c_value
                scvt[year_index] = arr

    c_tot = {}
    for year, values in scvt.items():
        c_tot[year] = np.average(values)

    p_tot = {}
    for year, values in treatment_data.items():
        p_tot[year] = np.average(values)

    if dump_dir is not None:
        figure, axis = plt.subplots(1, 3)
        figure.set_figheight(10)
        figure.set_figwidth(18)

        plot_carbon_trajectories(axis, 'Carbon stock (All Matches Treatment)', 1, treatment_data, project_start)
        plot_carbon_trajectories(axis, 'Carbon stock (All Matches Control)', 2, scvt, project_start)
        plot_carbon_stock(axis, p_tot, c_tot, project_start)

        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, os.path.splitext(pairs)[0] + "-carbon-stock.png")
        figure.savefig(out_path)

        # Now for all the pairs we create a GeoJSON for visualising
        smds = { "pair_id": [], "feature": [], "smd": [] }
        for pair_idx, pairs in enumerate(matches):
            matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

            linestrings = []
            for _, row in matches_df.iterrows():
                ls = Feature(geometry=LineString([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                linestrings.append(ls)
            
            gc = FeatureCollection(linestrings)
            out_path = os.path.join(dump_dir, os.path.splitext(pairs)[0] + "-pairs.geojson")

            with open(out_path, "w") as f:
                f.write(dumps(gc))

            points = []
            for _, row in matches_df.iterrows():
                ls = Feature(geometry=MultiPoint([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                points.append(ls)

            points_gc = FeatureCollection(points)
            out_path = os.path.join(dump_dir, os.path.splitext(pairs)[0] + "-pairs-points.geojson")

            with open(out_path, "w") as f:
                f.write(dumps(points_gc))
            
            # We now compute statistics for each pairing mainly looking at SMD
            mean_std = matches_df.agg(['mean', 'std'])
            for col in matches_df.columns:
                # only go from K to S so we don't double count
                if col[0] == "k":
                    treat_mean = mean_std[col]["mean"]
                    feature = "_".join(col.split("_")[1:])
                    control_col = "s_" + feature
                    control_mean = mean_std[control_col]["mean"]
                    treat_std = mean_std[col]["std"]
                    control_std = mean_std[control_col]["std"]
                    smd = (treat_mean - control_mean) / np.sqrt((treat_std ** 2 + control_std ** 2) / 2)
                    smd = round(abs(smd), 8)
                    smds["pair_id"].append(os.path.splitext(pairs)[0])
                    smds["feature"].append(feature)
                    smds["smd"].append(smd)
        smd_path = os.path.join(dump_dir, os.path.splitext(project_geojson_file)[0].split("/")[-1:][0] + "-smd.csv")
        df = pd.DataFrame.from_dict(smds)
        df.to_csv(smd_path)

    result = {}

    for year, value in p_tot.items():
        result[year] = (value - c_tot[year])

    with open(output_csv, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "additionality"])
        for year, result in result.items():
            writer.writerow([year, result])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes additionality for a range of years using pre-calculated pixel matches."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_file",
        help="GeoJSON files containing the polygons for the project's boundary",
    )
    parser.add_argument(
        "--project_start",
        type=int,
        required=True,
        dest="project_start",
        help="The start year of the project.",
    )
    parser.add_argument(
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Year of project evalation",
    )
    parser.add_argument(
        "--density",
        type=str,
        required=True,
        dest="carbon_density",
        help="The path the CSV containing carbon density values.",
    )
    parser.add_argument(
        "--matches",
        type=str,
        required=True,
        dest="matches",
        help="Directory containing the parquet files of the matches.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_csv",
        help="The destination output CSV path.",
    )

    args = parser.parse_args()

    generate_additionality(
        args.project_boundary_file,
        args.project_start,
        args.evaluation_year,
        args.carbon_density,
        args.matches,
        args.output_csv,
    )
