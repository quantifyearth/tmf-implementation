import glob
import os
import logging
import csv
import argparse
from functools import partial

import numpy as np
import pandas as pd
import geopandas as gpd
from yirgacheffe.layers import RasterLayer, VectorLayer, GroupLayer  # type: ignore

from methods.common import LandUseClass
from methods.common.geometry import area_for_geometry

EXPECTED_NUMBER_OF_MATCH_ITERATIONS = 100
MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def generate_leakage(
    project_geojson_file: str,
    leakage_geojson_file: str,
    project_start: str,
    end_year: int,
    jrc_directory_path: str,
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

    # We need pixel scales etc. to load in the vector for the project so we grab
    # one of the JRC tiles for this
    luc_for_metadata = glob.glob("*.tif", root_dir=jrc_directory_path)

    if len(luc_for_metadata) == 0:
        raise ValueError("No JRC TIF files found in the JRC directory")

    tif_for_metadata = RasterLayer.layer_from_file(
        os.path.join(jrc_directory_path, luc_for_metadata[0])
    )
    # project_boundary = VectorLayer.layer_from_file(
    #     project_geojson_file,
    #     None,
    #     tif_for_metadata.pixel_scale,
    #     tif_for_metadata.projection,
    # )
    leakage_zone = VectorLayer.layer_from_file(
        leakage_geojson_file,
        None,
        tif_for_metadata.pixel_scale,
        tif_for_metadata.projection,
    )
    total_pixels = leakage_zone.sum()


    # We calculate area using projections and not the inaccurate 30 * 30 approximation
    project_gpd = gpd.read_file(project_geojson_file)
    project_area_msq = area_for_geometry(project_gpd)
    leakage_gpd = gpd.read_file(leakage_geojson_file)
    leakage_area_msq = area_for_geometry(leakage_gpd)

    logging.info("Project area: %.2fmsq", project_area_msq)
    logging.info("Leakage area: %.2fmsq", leakage_area_msq)

    # TODO: see other TODO
    # total_pixels_project = project_boundary.sum()

    for year_index in range(project_start, end_year + 1):
        logging.info("Calculating leakage carbon for %i", year_index)

        # TODO: Double check with Michael this is the correct thing to do
        lucs = GroupLayer(
            [
                RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename))
                for filename in glob.glob(
                    f"*{year_index}*.tif", root_dir=jrc_directory_path
                )
            ]
        )

        # LUCs only in leakage zone
        intersection = RasterLayer.find_intersection([lucs, leakage_zone])
        leakage_zone.set_window_for_intersection(intersection)

        lucs.set_window_for_intersection(intersection)

        def is_in_class(class_, data):
            return np.where(data != class_, 0.0, 1.0)

        lucs_in_leakage = lucs * leakage_zone

        undisturbed = lucs_in_leakage.numpy_apply(
            partial(is_in_class, LandUseClass.UNDISTURBED)
        )
        degraded = lucs_in_leakage.numpy_apply(
            partial(is_in_class, LandUseClass.DEGRADED)
        )
        deforested = lucs_in_leakage.numpy_apply(
            partial(is_in_class, LandUseClass.DEFORESTED)
        )
        regrowth = lucs_in_leakage.numpy_apply(
            partial(is_in_class, LandUseClass.REGROWTH)
        )
        water = lucs_in_leakage.numpy_apply(partial(is_in_class, LandUseClass.WATER))
        other = lucs_in_leakage.numpy_apply(partial(is_in_class, LandUseClass.OTHER))

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

        # Quick Sanity Check
        prop = np.sum(proportions)
        assert 0.99 < prop < 1.01

        # TODO: the assumption of 30 x 30 resolution is not best practice
        areas = proportions * (leakage_area_msq / 10000)

        # Total carbon densities per class
        s_values = areas * density

        l_tot[year_index] = s_values.sum()

    matches = glob.glob("*.parquet", root_dir=matches_directory)

    assert len(matches) == EXPECTED_NUMBER_OF_MATCH_ITERATIONS

    scvt = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing leakage for %s", pairs)
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

            # TODO: Project area or leakage area?!
            areas_c = proportions_c * (leakage_area_msq / 10000)

            s_c = areas_c * density

            if scvt.get(year_index) is not None:
                scvt[year_index][pair_idx] = s_c.sum()
            else:
                arr = [0 for _ in range(EXPECTED_NUMBER_OF_MATCH_ITERATIONS)]
                arr[pair_idx] = s_c.sum()
                scvt[year_index] = arr

    c_tot = {}
    for year, values in scvt.items():
        c_tot[year] = np.average(values)

    result = {}

    for year, value in l_tot.items():
        result[year] = max(0, (value - c_tot[year]) * MOLECULAR_MASS_CO2_TO_C_RATIO)

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

    generate_leakage(
        args.project_boundary_file,
        args.project_leakage_zone,
        args.project_start,
        args.evaluation_year,
        args.jrc_directory_path,
        args.carbon_density,
        args.matches,
        args.output_csv,
    )
