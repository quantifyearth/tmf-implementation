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

from methods.common import LandUseClass, dump_dir
from methods.common.geometry import area_for_geometry

EXPECTED_NUMBER_OF_MATCH_ITERATIONS = 100
MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

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
    axis[0].set_title('Carbon stock (Treatment and Average Control)')
    axis[0].set_xlabel('Year')
    axis[0].set_ylabel('Carbon Stock (MgCO2e)')
    axis[0].legend(loc="lower left")

def plot_carbon_trajectories(axis, ts):
    x_axis = []
    y_axis = []
    for (k, v) in ts.items():
        x_axis.append(k)
        y_axis.append(ts[k])
    axis[1].plot(x_axis, y_axis)
    axis[1].set_title('Carbon stock (All Matches)')
    axis[1].set_xlabel('Year')
    axis[1].set_ylabel('Carbon Stock (MgCO2e)')

def generate_additionality(
    project_geojson_file: str,
    project_start: str,
    end_year: int,
    jrc_directory_path: str,
    carbon_density: str,
    matches_directory: str,
    output_csv: str,
) -> int:
    # Land use classes per year for the project
    p_tot = {}

    # TODO: may be present in config, in which case use that, but for now we use
    # the calculate version.
    density_df = pd.read_csv(carbon_density)
    density = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Density may have left some LUCs out like water
    for _, row in density_df.iterrows():
        luc = row["land use class"]
        density[int(luc) - 1] = row["carbon density"]

    # We need pixel scales etc. to load in the vector for the project so we grab
    # one of the JRC tiles for this
    luc_for_metadata = glob.glob("*.tif", root_dir=jrc_directory_path)

    if len(luc_for_metadata) == 0:
        raise ValueError("No JRC TIF files found in the JRC directory")

    tif_for_metadata = RasterLayer.layer_from_file(
        os.path.join(jrc_directory_path, luc_for_metadata[0])
    )

    project_boundary = VectorLayer.layer_from_file(
        project_geojson_file,
        None,
        tif_for_metadata.pixel_scale,
        tif_for_metadata.projection,
    )

    total_pixels = project_boundary.sum()

    # We calculate area using projections and not the inaccurate 30 * 30 approximation
    project_gpd = gpd.read_file(project_geojson_file)
    project_area_msq = area_for_geometry(project_gpd)

    logging.info("Project area: %.2fmsq", project_area_msq)

    for year_index in range(project_start, end_year + 1):
        logging.info("Calculating project carbon for %i", year_index)

        # TODO: Double check with Michael this is the correct thing to do
        lucs = GroupLayer(
            [
                RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename))
                for filename in glob.glob(
                    f"*{year_index}*.tif", root_dir=jrc_directory_path
                )
            ]
        )

        # LUCs only in project boundary
        intersection = RasterLayer.find_intersection([lucs, project_boundary])
        project_boundary.set_window_for_intersection(intersection)

        lucs.set_window_for_intersection(intersection)

        def is_in_class(class_, data):
            return np.where(data != class_, 0.0, 1.0)

        lucs_in_project = lucs * project_boundary

        undisturbed = lucs_in_project.numpy_apply(
            partial(is_in_class, LandUseClass.UNDISTURBED.value)
        )
        degraded = lucs_in_project.numpy_apply(
            partial(is_in_class, LandUseClass.DEGRADED.value)
        )
        deforested = lucs_in_project.numpy_apply(
            partial(is_in_class, LandUseClass.DEFORESTED.value)
        )
        regrowth = lucs_in_project.numpy_apply(
            partial(is_in_class, LandUseClass.REGROWTH.value)
        )
        water = lucs_in_project.numpy_apply(partial(is_in_class, LandUseClass.WATER.value))

        other = lucs_in_project.numpy_apply(partial(is_in_class, LandUseClass.OTHER.value))

        proportions = np.array(
            [
                undisturbed.sum() / total_pixels,
                degraded.sum() / total_pixels,
                deforested.sum() / total_pixels,
                regrowth.sum() / total_pixels,
                water.sum() / total_pixels,
                other.sum() / total_pixels,
            ]
        )

        logging.info(f"Proportions: {proportions}")

        # Quick Sanity Check
        prop = np.sum(proportions)
        assert 0.99 < prop < 1.01

        # Converts project_area_msg to ha
        areas = proportions * (project_area_msq / 10000)

        logging.info(f"Areas: {areas}")

        # Total carbon densities per class
        s_values = areas * density
        p_tot_value = s_values.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

        logging.info("Additionality is %f", p_tot_value)

        p_tot[year_index] = p_tot_value

    matches = glob.glob("*.parquet", root_dir=matches_directory)

    assert len(matches) == EXPECTED_NUMBER_OF_MATCH_ITERATIONS

    scvt = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing additionality for %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        for year_index in range(project_start, end_year + 1):
            total_pixels_c = len(matches_df)

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"luc_{year_index}"].value_counts()

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

    if dump_dir is not None:
        figure, axis = plt.subplots(1, 2)
        figure.set_figheight(10)
        figure.set_figwidth(15)

        plot_carbon_trajectories(axis, scvt)
        plot_carbon_stock(axis, p_tot, c_tot)

        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "1201-carbon-stock.png")
        figure.savefig(path)


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
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years.",
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
        args.jrc_directory_path,
        args.carbon_density,
        args.matches,
        args.output_csv,
    )
