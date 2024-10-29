import argparse
import glob
import os
import tempfile
from collections import namedtuple
from itertools import product
from typing import List, Optional

import pandas as pd
from geopandas import gpd  # type: ignore
from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer  # type: ignore
from yirgacheffe.window import PixelScale  # type: ignore

from methods.common import LandUseClass
from methods.common.geometry import area_for_geometry, expand_boundaries
from methods.common.luc import luc_range

HECTARE_WIDTH_IN_METERS = 100
PIXEL_WIDTH_IN_METERS = 30
SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.25
LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.05

# The '2 *' in this is because I'm just considering one axis, rather than area
# PIXEL_SKIP_SMALL_PROJECT = \
#     round((HECTARE_WIDTH_IN_METERS / (2 * SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)
# PIXEL_SKIP_LARGE_PROJECT = \
#     round((HECTARE_WIDTH_IN_METERS / (2 * LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)

PIXEL_SKIP_SMALL_PROJECT = 1
PIXEL_SKIP_LARGE_PROJECT = 1

MatchingCollection = namedtuple('MatchingCollection',
    ['boundary', 'lucs', 'cpcs', 'ecoregions', 'elevation', 'slope', 'access', 'countries'])

def build_layer_collection(
    pixel_scale: PixelScale,
    projection: str,
    luc_years: List[int],
    cpc_years: List[int],
    boundary_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
) -> MatchingCollection:
    # profiler = cProfile.Profile()
    # profiler.enable()
    outline_layer = VectorLayer.layer_from_file(boundary_filename, None, pixel_scale, projection)

    lucs = [
        GroupLayer([
            RasterLayer.layer_from_file(
                os.path.join(jrc_directory_path, filename)
            ) for filename in
                glob.glob(f"*{year}*.tif", root_dir=jrc_directory_path)
        ], name=f"luc_{year}") for year in luc_years
    ]

    cpcs = [
        GroupLayer([
            RasterLayer.layer_from_file(
                os.path.join(cpc_directory_path, filename)
            ) for filename in
                glob.glob(f"*{year_class[0]}*_{year_class[1].value}.tif", root_dir=cpc_directory_path)
        ], name=f"cpc_{year_class}") for year_class in product(cpc_years,
            [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED])
    ]

    # ecoregions is such a heavy layer it pays to just rasterize it once - we should possibly do this once
    # as part of import of the ecoregions data
    ecoregions = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(ecoregions_directory_path, filename)) for filename in
            glob.glob("*.tif", root_dir=ecoregions_directory_path)
    ], name="ecoregions")

    elevation = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(elevation_directory_path, filename)) for filename in
            glob.glob("srtm*.tif", root_dir=elevation_directory_path)
    ], name="elevation")
    slopes = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(slope_directory_path, filename)) for filename in
            glob.glob("slope*.tif", root_dir=slope_directory_path)
    ], name="slopes")

    access = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(access_directory_path, filename)) for filename in
            glob.glob("*.tif", root_dir=access_directory_path)
    ], name="access")

    countries = RasterLayer.layer_from_file(countries_raster_filename)

    # constrain everything to project boundaries
    layers = [elevation, slopes, ecoregions, access, countries] + lucs + cpcs
    for layer in layers:
        if layer.pixel_scale != pixel_scale:
            raise ValueError(f"Raster {layer.name} is at wrong pixel scale")
        layer.set_window_for_intersection(outline_layer.area)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()

    return MatchingCollection(
        boundary=outline_layer,
        lucs=lucs,
        cpcs=cpcs,
        ecoregions=ecoregions,
        elevation=elevation,
        slope=slopes,
        access=access,
        countries=countries,
    )

def calculate_k(
    project_boundary_filename: str,
    start_year: int,
    evaluation_year: int,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    buffer: Optional[int],
    result_dataframe_filename: str,
) -> None:
    project = gpd.read_file(project_boundary_filename)

    # Assumption: even if a buffer is specified, we use the unbuffered project
    # boundary as the project area to work out pixel_skip
    project_area_in_metres_squared = area_for_geometry(project)
    project_area_in_hectares = project_area_in_metres_squared / 10_000
    pixel_skip = PIXEL_SKIP_LARGE_PROJECT if (project_area_in_hectares > 250_000) else PIXEL_SKIP_SMALL_PROJECT

    with tempfile.TemporaryDirectory() as tmpdir:
        if buffer is not None:
            expanded_project = expand_boundaries(project, buffer)
            expanded_project_filename = os.path.join(tmpdir, "expanded-boundaries.geojson")
            expanded_project.to_file(expanded_project_filename, driver="GeoJSON")
            project_boundary_to_use = expanded_project_filename
        else:
            project_boundary_to_use = project_boundary_filename

        # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
        example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
        example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

        project_collection = build_layer_collection(
            example_jrc_layer.pixel_scale,
            example_jrc_layer.projection,
            list(luc_range(start_year, evaluation_year)),
            [start_year, start_year - 5, start_year - 10],
            project_boundary_to_use,
            jrc_directory_path,
            cpc_directory_path,
            ecoregions_directory_path,
            elevation_directory_path,
            slope_directory_path,
            access_directory_path,
            countries_raster_filename,
        )

        results = []

        project_width = project_collection.boundary.window.xsize
        for yoffset in range(0, project_collection.boundary.window.ysize, pixel_skip):
            row_boundary = project_collection.boundary.read_array(0, yoffset, project_width, 1)
            row_elevation = project_collection.elevation.read_array(0, yoffset, project_width, 1)
            row_ecoregion = project_collection.ecoregions.read_array(0, yoffset, project_width, 1)
            row_slope = project_collection.slope.read_array(0, yoffset, project_width, 1)
            row_access = project_collection.access.read_array(0, yoffset, project_width, 1)
            row_countries = project_collection.countries.read_array(0, yoffset, project_width, 1)
            row_luc = [
                luc.read_array(0, yoffset, project_width, 1) for luc in project_collection.lucs
            ]
            row_cpcs = [
                cpc.read_array(0, yoffset, project_width, 1) for cpc in project_collection.cpcs
            ]

            for xoffset in range(0, project_width, pixel_skip):
                if not row_boundary[0][xoffset]:
                    continue
                lucs = [x[0][xoffset] for x in row_luc]
                cpcs = [x[0][xoffset] for x in row_cpcs]

                coord = project_collection.boundary.latlng_for_pixel(xoffset, yoffset)

                results.append([
                    xoffset,
                    yoffset,
                    coord[0],
                    coord[1],
                    row_elevation[0][xoffset],
                    row_slope[0][xoffset],
                    row_ecoregion[0][xoffset],
                    row_access[0][xoffset],
                    row_countries[0][xoffset],
                ] + lucs + cpcs)

        luc_columns = [f'luc_{year}' for year in luc_range(start_year, evaluation_year)]
        cpc_columns = ['cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']
        output = pd.DataFrame(
            results,
            columns=['x', 'y', 'lat', 'lng', 'elevation', 'slope', 'ecoregion', 'access', 'country'] \
                + luc_columns + cpc_columns
        )
        output.to_parquet(result_dataframe_filename)

def main():
    parser = argparse.ArgumentParser(description="Calculates sample pixels in project, aka set K")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_boundary_filename",
        help="GeoJSON File of project boundary."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        dest="start_year",
        help="Year project started."
    )
    parser.add_argument(
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Year of project evalation"
    )
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--cpc",
        type=str,
        required=True,
        dest="cpc_directory_path",
        help="Directory containing Coarsened Proportional Coverage GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        dest="ecoregions_directory_path",
        help="Directory containing Ecoregions GeoTIFF tiles."
    )
    parser.add_argument(
        "--elevation",
        type=str,
        required=True,
        dest="elevation_directory_path",
        help="Directory containing SRTM elevation GeoTIFF tiles."
    )
    parser.add_argument(
        "--slope",
        type=str,
        required=True,
        dest="slope_directory_path",
        help="Directory containing slope GeoTIFF tiles."
    )
    parser.add_argument(
        "--access",
        type=str,
        required=True,
        dest="access_directory_path",
        help="Directory containing access to health care GeoTIFF tiles."
    )
    parser.add_argument(
        "--countries-raster",
        type=str,
        required=True,
        dest="countries_raster_filename",
        help="GeoJSON of country boundaries."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination parquet file for results."
    )
    parser.add_argument(
        "--buffer",
        type=int,
        required=False,
        dest="buffer",
        help="The size, in metres, of the buffer to apply to the edge of the project boundary."
    )
    args = parser.parse_args()

    calculate_k(
        args.project_boundary_filename,
        args.start_year,
        args.evaluation_year,
        args.jrc_directory_path,
        args.cpc_directory_path,
        args.ecoregions_directory_path,
        args.elevation_directory_path,
        args.slope_directory_path,
        args.access_directory_path,
        args.countries_raster_filename,
        args.buffer,
        args.output_filename
    )

if __name__ == "__main__":
    main()
