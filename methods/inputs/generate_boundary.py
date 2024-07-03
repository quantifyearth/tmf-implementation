import argparse
import sys

from geopandas import gpd # type: ignore

from methods.common.geometry import expand_boundaries

PROJECT_BOUNDARY_RADIUS_IN_METRES = 30_000

def generate_boundary(input_filename: str, output_filename: str) -> None:
    project_boundaries = gpd.read_file(input_filename)
    result = expand_boundaries(project_boundaries, PROJECT_BOUNDARY_RADIUS_IN_METRES)
    result.to_file(output_filename, driver="GeoJSON")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generates expanded boundary shape for a project")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_filename",
        help="GeoJSON File of project boundary."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="GeoJSON File of project expanded boundary."
    )
    args = parser.parse_args()

    try:
        generate_boundary(args.project_boundary_filename, args.output_filename)
    except FileNotFoundError as exc:
        print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
