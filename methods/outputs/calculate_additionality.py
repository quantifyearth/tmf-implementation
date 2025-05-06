import csv
import os
import sys
import argparse
import logging

import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore

from methods.common import LandUseClass
from methods.common.additionality import generate_additionality
from methods.common.geometry import area_for_geometry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes additionality for a range of years using pre-calculated pixel matches."
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_file",
        help="GeoJSON file containing the polygons for the project's boundary",
    )
    parser.add_argument(
        "--project_start",
        type=str,
        required=True,
        dest="project_start",
        help="The start year of the project.",
    )
    parser.add_argument(
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Final year of project evaluation for the analysis range.",
    )
    parser.add_argument(
        "--density",
        type=str,
        required=True,
        dest="carbon_density",
        help="The path the CSV or Parquet containing carbon density values.",
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

    logging.info(f"Loading carbon density from {args.carbon_density}")
    _, ext = os.path.splitext(args.carbon_density)
    if ext == ".csv":
        density_df = pd.read_csv(args.carbon_density)
    elif ext == ".parquet":
        density_df = pd.read_parquet(args.carbon_density)
    else:
        logging.error(f"Unrecognised file extension for density file: {ext}")
        sys.exit(1)

    density = np.zeros(len(LandUseClass))
    logging.info(f"Populating density array (size: {len(density)})")
    for _, row in density_df.iterrows():
        try:
            luc_val = int(row["land use class"])
            carbon_val = float(row["carbon density"])
            if 1 <= luc_val <= len(density):
                density[luc_val - 1] = carbon_val
            else:
                logging.warning(f"Skipping row with invalid LUC value {luc_val} in density file.")
        except (ValueError, KeyError, TypeError) as e:
            logging.warning(f"Skipping invalid row in density file: {row}. Error: {e}")

    logging.info(f"Loading project boundary from {args.project_boundary_file}")
    project_gpd = gpd.read_file(args.project_boundary_file)
    project_area_msq = area_for_geometry(project_gpd)
    logging.info(f"Calculated project area: {project_area_msq:.2f} m^2")

    logging.info("Generating additionality results...")
    results_df = generate_additionality(
        project_area_msq=project_area_msq,
        project_start=args.project_start,
        end_year=args.evaluation_year,
        density=density,
        matches_directory=args.matches,
    )

    logging.info(f"Saving additionality results to {args.output_csv}")
    try:
        results_df.to_csv(args.output_csv, index=False, float_format='%.6f')
        logging.info("--Additionality calculated.--")
    except Exception as e:
        logging.error(f"Failed to save results CSV: {e}")
        sys.exit(1)
