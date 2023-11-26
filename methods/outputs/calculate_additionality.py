import csv
import argparse

from methods.common.additionality import generate_additionality

EXPECTED_NUMBER_OF_ITERATIONS = 100

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

    add = generate_additionality(
        args.project_boundary_file,
        args.project_start,
        args.evaluation_year,
        args.carbon_density,
        args.matches,
        EXPECTED_NUMBER_OF_ITERATIONS
    )

    with open(args.output_csv, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "additionality"])
        for year, value in add.items():
            writer.writerow([year, value])
