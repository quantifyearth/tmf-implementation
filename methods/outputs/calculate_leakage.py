import csv
import os
import sys
import argparse

import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore

from methods.common.additionality import generate_additionality
from methods.common.geometry import area_for_geometry

EXPECTED_NUMBER_OF_ITERATIONS = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes leakage for a range of years using pre-calculated pixel matches."
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

    # TODO: may be present in config, in which case use that, but for now we use
    # the calculate version.
    _, ext = os.path.splitext(args.carbon_density)
    if ext == ".csv":
        density_df = pd.read_csv(args.carbon_density)
    elif ext == ".parquet":
        density_df = pd.read_parquet(args.carbon_density)
    else:
        print(f"Unrecognised file extension: {ext}", file=sys.stderr)
        sys.exit(1)
    density = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Density may have left some LUCs out like water
    for _, row in density_df.iterrows():
        luc = row["land use class"]
        density[int(luc) - 1] = row["carbon density"]

    leakage_gpd = gpd.read_file(args.project_leakage_zone)
    leakage_area_msq = area_for_geometry(leakage_gpd)

    leakage = generate_additionality(
        leakage_area_msq,
        args.project_start,
        args.evaluation_year,
        density,
        args.matches,
        EXPECTED_NUMBER_OF_ITERATIONS
    )

    with open(args.output_csv, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "leakage"])
        for year, value in leakage.items():
            writer.writerow([year, value])