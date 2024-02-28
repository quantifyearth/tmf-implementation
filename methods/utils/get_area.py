import argparse

import geopandas as gpd # type: ignore

from methods.common.geometry import area_for_geometry

def main() -> None:
    parser = argparse.ArgumentParser(description="Prints area of area in m^2")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_filename",
        help="GeoJSON File of a shape."
    )
    parser.add_argument(
        "--hectares",
        required=False,
        default=False,
        dest="in_hectares",
        help="Convert to hectares",
        action='store_true'
    )
    args = parser.parse_args()

    project_gpd = gpd.read_file(args.project_boundary_filename)
    project_area_msq = area_for_geometry(project_gpd)

    if args.in_hectares:
        print(f"{project_area_msq / 10_000}")
    else:
        print(f"{project_area_msq}")


if __name__ == "__main__":
    main()
