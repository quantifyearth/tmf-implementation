import glob
import os
import logging
import csv
import argparse
from functools import partial

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from yirgacheffe.layers import RasterLayer, VectorLayer, GroupLayer  # type: ignore
from geojson import LineString, FeatureCollection, Feature, MultiPoint, dumps # type: ignore

from methods.common import LandUseClass, dump_dir
from methods.common.geometry import area_for_geometry

EXPECTED_NUMBER_OF_MATCH_ITERATIONS = 100
MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def plot_carbon_stock(axis, arr, cf):
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
    axis[0].legend(loc="lower left")

def plot_carbon_trajectories(axis, title, idx, ts):
    x_axis = []
    y_axis = []
    for (k, v) in ts.items():
        x_axis.append(k)
        y_axis.append(ts[k])
    axis[idx].plot(x_axis, y_axis)
    axis[idx].set_title(title)
    axis[idx].set_xlabel('Year')
    axis[idx].set_ylabel('Carbon Stock (MgCO2e)')

def generate_leakage(
    project_geojson_file: str,
    leakage_geojson_file: str,
    project_start: str,
    end_year: int,
    carbon_density: str,
    matches_directory: str,
    output_csv: str,
) -> int:
    # Make linter happy whilst we decide if we need to project polygon or not
    logging.info("Generating leakage for %s", project_geojson_file)

    # Land use classes per year for the leakage zone
    l_tot = {}

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
    leakage_gpd = gpd.read_file(leakage_geojson_file)
    leakage_area_msq = area_for_geometry(leakage_gpd)

    logging.info("Project area: %.2fmsq", project_area_msq)
    logging.info("Leakage area: %.2fmsq", leakage_area_msq)

    matches = glob.glob("*.parquet", root_dir=matches_directory)
    assert len(matches) == EXPECTED_NUMBER_OF_MATCH_ITERATIONS

    l_tot = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing leakage for %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        for year_index in range(project_start, end_year + 1):
            total_pixels = len(matches_df)

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"k_luc_{year_index}"].value_counts()

            for luc in value_count_year.index.tolist():
                values[int(luc) - 1] = value_count_year[luc]

            undisturbed = values[LandUseClass.UNDISTURBED - 1]
            degraded = values[LandUseClass.DEGRADED - 1]
            deforested = values[LandUseClass.DEFORESTED - 1]
            regrowth = values[LandUseClass.REGROWTH - 1]
            water = values[LandUseClass.WATER - 1]
            other = values[LandUseClass.OTHER - 1]

            proportions = np.array(
                [
                    undisturbed / total_pixels,
                    degraded / total_pixels,
                    deforested / total_pixels,
                    regrowth / total_pixels,
                    water / total_pixels,
                    other / total_pixels,
                ]
            )

            areas = proportions * (leakage_area_msq / 10000)

            s = areas * density

            s_value = s.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            if l_tot.get(year_index) is not None:
                l_tot[year_index][pair_idx] = s_value
            else:
                arr = [0 for _ in range(EXPECTED_NUMBER_OF_MATCH_ITERATIONS)]
                arr[pair_idx] = s_value
                l_tot[year_index] = arr

    scvt = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing leakage for %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        for year_index in range(project_start, end_year + 1):
            total_pixels_c = len(matches_df)

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"s_luc_{year_index}"].value_counts()

            for luc in value_count_year.index.tolist():
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

            areas_c = proportions_c * (leakage_area_msq / 10000)

            s_c = areas_c * density

            s_c_value = s_c.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            if scvt.get(year_index) is not None:
                scvt[year_index][pair_idx] = s_c_value
            else:
                arr = [0 for _ in range(EXPECTED_NUMBER_OF_MATCH_ITERATIONS)]
                arr[pair_idx] = s_c_value
                scvt[year_index] = arr

    c_tot = {}
    for year, values in scvt.items():
        c_tot[year] = np.average(values)

    project = {}
    for year, values in l_tot.items():
        project[year] = np.average(values)

    result = {}

    if dump_dir is not None:
        figure, axis = plt.subplots(1, 3)
        figure.set_figheight(10)
        figure.set_figwidth(15)

        plot_carbon_trajectories(axis, 'Carbon stock (All Matches Treatment)', 1, l_tot)
        plot_carbon_trajectories(axis, 'Carbon stock (All Matches Control)', 2, scvt)
        plot_carbon_stock(axis, project, c_tot)

        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "1201-leakage-carbon-stock.png")
        figure.savefig(path)

        # Now for all the pairs we create a GeoJSON for visualising
        for pair_idx, pairs in enumerate(matches):
            matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

            linestrings = []
            for _, row in matches_df.iterrows():
                ls = Feature(geometry=LineString([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                linestrings.append(ls)
            
            points = []
            for _, row in matches_df.iterrows():
                ls = Feature(geometry=MultiPoint([(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]))
                points.append(ls)
            
            gc = FeatureCollection(linestrings)
            out_path = os.path.join(dump_dir, os.path.splitext(pairs)[0] + "-leakage-pairs.geojson")

            with open(out_path, "w") as f:
                f.write(dumps(gc))

            points_gc = FeatureCollection(points)
            out_path = os.path.join(dump_dir, os.path.splitext(pairs)[0] + "-leakage-pairs-points.geojson")

            with open(out_path, "w") as f:
                f.write(dumps(points_gc))

    for year, value in project.items():
        result[year] = max(0, (value - c_tot[year]))

    with open(output_csv, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "leakage"])
        for year, result in result.items():
            writer.writerow([year, result])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes leakage for a range of years using pre-calculated pixel matches."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_file",
        help="GeoJSON files containing the polygons for the project's boundary",
    )
    parser.add_argument(
        "--leakage_zone",
        type=str,
        required=True,
        dest="project_leakage_zone",
        help="GeoJSON files containing the polygons for the project's leakage zone",
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

    generate_leakage(
        args.project_boundary_file,
        args.project_leakage_zone,
        args.project_start,
        args.evaluation_year,
        args.carbon_density,
        args.matches,
        args.output_csv,
    )
