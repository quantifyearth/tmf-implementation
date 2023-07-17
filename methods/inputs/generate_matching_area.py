import glob
import os
import sys

import shapely # type: ignore
from fiona.errors import DriverError # type: ignore
from geopandas import gpd # type: ignore

from methods.common.geometry import expand_boundaries

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

MATCHING_RADIUS_IN_METRES = 2_000_000
LEAKAGE_BUFFER_IN_METRES = 5_000

def generate_matching_area(
    project_shape_filename: str,
    country_iso_a2_code: str,
    countries_shape_filename: str,
    ecoregions_shape_filename: str,
    other_projects_directory: str,
    output_shape_filename: str
) -> None:
    project_boundaries = gpd.read_file(project_shape_filename)
    extended_project = expand_boundaries(project_boundaries, MATCHING_RADIUS_IN_METRES)
    unified_extended_boundary = shapely.unary_union(extended_project)

    country_shapes = gpd.read_file(countries_shape_filename)
    if country_shapes.crs != project_boundaries.crs:
        country_shapes = country_shapes.to_crs(project_boundaries.crs)
    if len(country_iso_a2_code) != 2:
        raise ValueError(f"Expected country code to be 2 letters long, got {country_iso_a2_code}")
    country = country_shapes[country_shapes['ISO_A2'] == country_iso_a2_code]
    if country.shape[0] == 0:
        raise ValueError(f"Got no matches for country ISO A2 code {country_iso_a2_code}")
    elif country.shape[0] > 1:
        raise ValueError(f"Got too many matches for country ISO A2 code {country_iso_a2_code}")

    ecoregions = gpd.read_file(ecoregions_shape_filename)
    if ecoregions.crs != project_boundaries.crs:
        ecoregions = ecoregions.to_crs(project_boundaries.crs)

    # Intersects requires a single geometry, and a project is a list of polygons usually, so
    # need to make a single multipolygon
    project_multipolygon = shapely.geometry.MultiPolygon([polygon for polygon in project_boundaries.geometry])
    overlapping_ecoregions = ecoregions[ecoregions.intersects(project_multipolygon)]
    if overlapping_ecoregions.shape[0] == 0:
        raise ValueError("No overlapping ecoregions found")
    unified_ecoregions = shapely.unary_union(overlapping_ecoregions.geometry)

    # Now we have:
    # * The expanded project boundary
    # * The country area
    # * The area of overlapping ecoregions
    # And now we need the intersection of all three (before we then subtract things)
    step_1 = shapely.intersection(unified_extended_boundary, country.geometry.values[0])
    step_2 = shapely.intersection(step_1, unified_ecoregions)

    # Now remove the project plus leakage buffer
    project_leakage = expand_boundaries(project_boundaries, LEAKAGE_BUFFER_IN_METRES)
    step_3 = shapely.difference(step_2, project_leakage.geometry)

    # Now for each other project, remove it and a leakage buffer
    for filename in glob.glob("*.geojson", root_dir=other_projects_directory):
        other_project = gpd.read_file(os.path.join(other_projects_directory, filename))
        extended_other_project = expand_boundaries(other_project, LEAKAGE_BUFFER_IN_METRES)
        step_3 = shapely.difference(step_3, extended_other_project.geometry)

    result = gpd.GeoDataFrame.from_features(gpd.GeoSeries(step_3), crs=project_boundaries.crs)
    result.to_file(output_shape_filename, driver="GeoJSON")


if __name__ == "__main__":
    try:
        project_shape_filename = sys.argv[1]
        country_iso_a2_code = sys.argv[2]
        countries_shape_filename = sys.argv[3]
        ecoregions_shape_filename = sys.argv[4]
        other_projects_directory = sys.argv[5]
        output_shape_filename = sys.argv[6]
    except IndexError:
        print(f"Usage: {sys.argv[0]} PROJECT_SHAPEFILE COUNTRY_CODE COUNTRY_SHAPEFILE ECOREGIONS_SHAPEFILE OTHER_PROJECTS_DIRECTORY OUTPUT_SHAPEFILE", file=sys.stderr)
        sys.exit(1)

    try:
        generate_matching_area(
            project_shape_filename,
            country_iso_a2_code,
            countries_shape_filename,
            ecoregions_shape_filename,
            other_projects_directory,
            output_shape_filename
        )
    except FileNotFoundError as exc:
        print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
        sys.exit(1)
    except DriverError as exc:
        print(exc.args[0], file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Bad value: {exc.args[0]}", file=sys.stderr)
        sys.exit(1)

