import argparse
import glob
import json
import os
import sys
import traceback

import shapely # type: ignore
from fiona.errors import DriverError # type: ignore
from geopandas import gpd # type: ignore

from methods.common.geometry import expand_boundaries
from methods.inputs.generate_leakage import LEAKAGE_BUFFER_IN_METRES

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

MATCHING_RADIUS_IN_METRES = 2_000_000

def generate_matching_area(
    project_shape_filename: str,
    country_iso_a2_code_filename: str,
    countries_shape_filename: str,
    ecoregions_shape_filename: str,
    other_projects_directory: str,
    output_shape_filename: str
) -> None:
    project_boundaries = gpd.read_file(project_shape_filename)
    extended_project = expand_boundaries(project_boundaries, MATCHING_RADIUS_IN_METRES)
    unified_extended_boundary = shapely.unary_union(extended_project)

    with open(country_iso_a2_code_filename, 'r') as codesfs:
        country_codes_list = json.loads(codesfs.read())

    print("Loading country shapes...")
    country_shapes = gpd.read_file(countries_shape_filename)
    if country_shapes.crs != project_boundaries.crs:
        print("Reprojecting country shapes")
        country_shapes = country_shapes.to_crs(project_boundaries.crs)
    print("Done loading country shapes")

    countries = country_shapes[country_shapes['ISO_A2'].isin(country_codes_list)]
    if countries.shape[0] == 0:
        raise ValueError(f"Got no matches for country ISO A2 codes {country_codes_list}")
    simplified_countries = shapely.unary_union(countries.geometry)

    print("Loading ecoregions...")
    ecoregions = gpd.read_file(ecoregions_shape_filename)
    if ecoregions.crs != project_boundaries.crs:
        print("Reprojecting ecoregions")
        ecoregions = ecoregions.to_crs(project_boundaries.crs)
    print("Done loading ecoregions")

    # Intersects requires a single geometry, and a project is a list of polygons usually, so
    # need to make a single multipolygon
    if project_boundaries.type[0] == "MultiPolygon":
        project_multipolygon = project_boundaries.geometry[0]
    else:
        project_multipolygon = shapely.geometry.MultiPolygon(polygon for polygon in project_boundaries.geometry)
    overlapping_ecoregions = ecoregions[ecoregions.intersects(project_multipolygon)]
    if overlapping_ecoregions.shape[0] == 0:
        raise ValueError("No overlapping ecoregions found")
    unified_ecoregions = shapely.unary_union(overlapping_ecoregions.geometry)

    # Now we have:
    # * The expanded project boundary
    # * The country area
    # * The area of overlapping ecoregions
    # And now we need the intersection of all three (before we then subtract things)
    step_1 = shapely.intersection(unified_extended_boundary, simplified_countries)
    step_2 = shapely.intersection(step_1, unified_ecoregions)

    # Now remove the project plus leakage buffer
    print("Removing project")
    project_leakage = expand_boundaries(project_boundaries, LEAKAGE_BUFFER_IN_METRES)
    step_3 = shapely.difference(step_2, project_leakage.geometry)

    # Now for each other project, remove it and a leakage buffer
    print("Removing other projects")
    for filename in glob.glob("*.geojson", root_dir=other_projects_directory):
        print(f"\tRemoving {filename}...")
        other_project = gpd.read_file(os.path.join(other_projects_directory, filename))

        # Some of the projects have quite complex boundaries (e.g., 2363), and so we
        # do an initial check just based on the simplified bounary to see if we need to
        # do the hard work.
        print("\t\tsimplifying...")
        outline = shapely.convex_hull(other_project.geometry)
        outline_dataframe = gpd.GeoDataFrame.from_features(gpd.GeoSeries(outline), crs=other_project.crs)
        extended_outline = expand_boundaries(outline_dataframe, LEAKAGE_BUFFER_IN_METRES)

        if not shapely.disjoint(step_3, extended_outline.geometry).any():
            print("\t\tExtending...")
            extended_other_project = expand_boundaries(other_project, LEAKAGE_BUFFER_IN_METRES)
            print("\t\tRemoving...")
            step_3 = shapely.difference(step_3, extended_other_project.geometry)
        else:
            print("\t\term")

    result = gpd.GeoDataFrame.from_features(gpd.GeoSeries(step_3), crs=project_boundaries.crs)
    result.to_file(output_shape_filename, driver="GeoJSON")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generates pixel matching area")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_shape_filename",
        help="GeoJSON File of project boundary."
    )
    parser.add_argument(
        "--countrycodes",
        type=str,
        required=True,
        dest="country_list_filename",
        help="JSON list of ISO A2 country names that project/leakage overlaps."
    )
    parser.add_argument(
        "--countries",
        type=str,
        required=True,
        dest="countries_shape_filename",
        help="GeoJSON of country boundaries."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        dest="ecoregions_shape_filename",
        help="GeoJSON of ecoregion boundaries."
    )
    parser.add_argument(
        "--projects",
        type=str,
        required=True,
        dest="other_projects_directory",
        help="Directory of other projects GeoJSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination GeoJSON file for results."
    )
    args = parser.parse_args()

    try:
        generate_matching_area(
            args.project_shape_filename,
            args.country_list_filename,
            args.countries_shape_filename,
            args.ecoregions_shape_filename,
            args.other_projects_directory,
            args.output_filename
        )
    except FileNotFoundError as exc:
        print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
        sys.exit(1)
    except DriverError as exc:
        print(exc.args[0], file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Bad value: {exc.args[0]}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
