import argparse
import glob
import os
import random
import tempfile
import logging
import multiprocessing
from functools import partial
from collections import namedtuple
from itertools import product
from typing import List, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer
from yirgacheffe.window import PixelScale

from methods.common import LandUseClass # Assuming this exists in your project structure
from methods.common.geometry import area_for_geometry, expand_boundaries # Assuming this exists
from methods.common.luc import luc_range # Assuming this exists

# --- Constants ---
HECTARE_WIDTH_IN_METERS = 100
PIXEL_WIDTH_IN_METERS = 30
# Adjust densities as needed
SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.05
LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.01

# Pixel skip calculation based on density
PIXEL_SKIP_SMALL_PROJECT = round((HECTARE_WIDTH_IN_METERS / (2 * SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)
PIXEL_SKIP_LARGE_PROJECT = round((HECTARE_WIDTH_IN_METERS / (2 * LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)

# --- Data Structures ---
MatchingCollection = namedtuple('MatchingCollection',
    ['boundary', 'lucs', 'fccs', 'ecoregions', 'elevation', 'slope', 'access', 'countries'])

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(processName)s %(message)s")

# --- Core Functions ---

def build_layer_collection(
    pixel_scale: PixelScale,
    projection: str,
    luc_years: List[int],
    fcc_years: List[int],
    boundary_filename: str,
    jrc_directory_path: str,
    fcc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
) -> MatchingCollection:
    """Builds the collection of layers needed for sampling."""
    outline_layer = VectorLayer.layer_from_file(boundary_filename, None, pixel_scale, projection)

    lucs = [
        GroupLayer([
            RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename)) for filename in
                glob.glob(f"*{year}*.tif", root_dir=jrc_directory_path)
        ], name=f"luc_{year}") for year in luc_years
    ]

    fccs = [
        GroupLayer([
            RasterLayer.layer_from_file(
                os.path.join(fcc_directory_path, filename)
            ) for filename in
                glob.glob(f"*{year_class[0]}*_{year_class[1].value}.tif", root_dir=fcc_directory_path)
        ], name=f"fcc_{year_class}") for year_class in product(fcc_years,
            [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED]) # Assuming LandUseClass enum
    ]

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

    # Constrain layers
    layers = [elevation, slopes, ecoregions, access, countries] + lucs + fccs
    for layer in layers:
        if hasattr(layer, 'pixel_scale') and layer.pixel_scale != pixel_scale: # Check if attribute exists
            logging.warning(f"Raster {getattr(layer, 'name', 'Unnamed')} might be at wrong pixel scale (Expected: {pixel_scale}, Got: {layer.pixel_scale}). Ensure inputs match or reproject.")
            # Add reprojection logic if necessary, or ensure inputs match
        if hasattr(layer, 'set_window_for_intersection') and hasattr(outline_layer, 'area'): # Check methods/attributes
             layer.set_window_for_intersection(outline_layer.area)

    return MatchingCollection(
        boundary=outline_layer, lucs=lucs, fccs=fccs, ecoregions=ecoregions,
        elevation=elevation, slope=slopes, access=access, countries=countries,
    )

def calculate_single_k_parquet(
    project_boundary_filename: str,
    start_year: int,
    evaluation_year: int,
    jrc_directory_path: str,
    fcc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    buffer_distance: int,
    result_parquet_filename: str,
    pixel_skip: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """Calculates and saves the K set (sample points) for a single offset to a Parquet file."""

    # Determine which boundary file to use initially
    # We always read the original project boundary first
    project_boundary_to_use = project_boundary_filename

    with tempfile.TemporaryDirectory() as tmpdir:
        # Apply buffer dynamically if buffer_distance > 0
        if buffer_distance > 0:
            project_gdf = gpd.read_file(project_boundary_filename) # Read original to apply buffer
            expanded_project = expand_boundaries(project_gdf, buffer_distance)
            expanded_project_filename = os.path.join(tmpdir, f"expanded_{os.path.basename(project_boundary_filename)}")
            expanded_project.to_file(expanded_project_filename, driver="GeoJSON")
            project_boundary_to_use = expanded_project_filename # Use the buffered file for sampling
            logging.debug(f"Applied buffer of {buffer_distance}m, using {project_boundary_to_use} for sampling.")
        else:
            logging.debug(f"Buffer is 0, using original boundary {project_boundary_to_use} for sampling.")

        # Get scale/projection from JRC
        try:
            example_jrc_filename = glob.glob(os.path.join(jrc_directory_path, "*.tif"))[0]
        except IndexError:
            raise FileNotFoundError(f"No TIF files found in JRC directory: {jrc_directory_path}")
        example_jrc_layer = RasterLayer.layer_from_file(example_jrc_filename)

        project_collection = build_layer_collection(
            example_jrc_layer.pixel_scale, example_jrc_layer.projection,
            list(luc_range(start_year, evaluation_year)),
            [start_year, start_year - 5, start_year - 10], # Assuming these FCC years
            project_boundary_to_use, jrc_directory_path, fcc_directory_path,
            ecoregions_directory_path, elevation_directory_path, slope_directory_path,
            access_directory_path, countries_raster_filename,
        )

        results = []
        project_width = project_collection.boundary.window.xsize
        project_height = project_collection.boundary.window.ysize

        for yoff in range(y_offset, project_height, pixel_skip):
            # Read row data efficiently
            row_boundary = project_collection.boundary.read_array(0, yoff, project_width, 1)
            if not row_boundary.any(): # Skip empty rows
                 continue
            row_elevation = project_collection.elevation.read_array(0, yoff, project_width, 1)
            row_ecoregion = project_collection.ecoregions.read_array(0, yoff, project_width, 1)
            row_slope = project_collection.slope.read_array(0, yoff, project_width, 1)
            row_access = project_collection.access.read_array(0, yoff, project_width, 1)
            row_countries = project_collection.countries.read_array(0, yoff, project_width, 1)
            row_luc_data = [luc.read_array(0, yoff, project_width, 1) for luc in project_collection.lucs]
            row_fcc_data = [fcc.read_array(0, yoff, project_width, 1) for fcc in project_collection.fccs]

            for xoff in range(x_offset, project_width, pixel_skip):
                if row_boundary[0, xoff]: # Check if pixel is within boundary
                    lucs = [luc_data[0, xoff] for luc_data in row_luc_data]
                    fccs = [fcc_data[0, xoff] for fcc_data in row_fcc_data]
                    coord = project_collection.boundary.latlng_for_pixel(xoff, yoff)

                    results.append([
                        xoff, yoff, coord[0], coord[1], # lat, lng
                        row_elevation[0, xoff], row_slope[0, xoff],
                        row_ecoregion[0, xoff], row_access[0, xoff],
                        row_countries[0, xoff],
                    ] + lucs + fccs)

        if not results:
            logging.warning(f"No points generated for offset ({x_offset}, {y_offset}). Check project boundary and data overlap.")
            return # Avoid creating empty parquet

        # Define column names
        luc_columns = [f'luc_{year}' for year in luc_range(start_year, evaluation_year)]
        # --- Use relative year offsets (0, 5, 10) for fcc column names ---
        fcc_years_relative = [0, 5, 10]
        fcc_column_names = [f"fcc{rel_year}_{'u' if luc_class == LandUseClass.UNDISTURBED else 'd'}"
                            for rel_year, luc_class in product(fcc_years_relative,
                                                               [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED])]

        output_df = pd.DataFrame(
            results,
            columns=['x', 'y', 'lat', 'lng', 'elevation', 'slope', 'ecoregion', 'access', 'country']
                      + luc_columns + fcc_column_names
        )
        output_df.to_parquet(result_parquet_filename)

def process_single_grid_task(
    i: int, # Grid index (0 to num_grids-1)
    args: argparse.Namespace,
    pixel_skip: int,
    x_offsets: List[int], # Now expects a list of x offsets
    y_offsets: List[int], # Now expects a list of y offsets
    output_directory: str,
    geojsons_directory: str
) -> Optional[int]:
    """
    Worker function for multiprocessing. Generates one Parquet grid
    and its corresponding GeoJSON file.
    """
    grid_index_1_based = i + 1
    parquet_filename = os.path.join(output_directory, f"k_{grid_index_1_based}.parquet")
    geojson_filename = os.path.join(geojsons_directory, f"k_{grid_index_1_based}.geojson")
    # Get the specific offset pair for this task index
    x_offset = x_offsets[i]
    y_offset = y_offsets[i]

    try:
        # 1. Generate Parquet file
        logging.info(f"Starting grid {grid_index_1_based} (offset {x_offset},{y_offset})...")
        calculate_single_k_parquet(
            args.project_boundary_filename, args.start_year, args.evaluation_year,
            args.jrc_directory_path, args.fcc_directory_path, args.ecoregions_directory_path,
            args.elevation_directory_path, args.slope_directory_path, args.access_directory_path,
            args.countries_raster_filename, args.buffer, parquet_filename, # Pass buffer distance (int) here
            pixel_skip, x_offset, y_offset # Pass pixel_skip here
        )
        logging.info(f"Generated grid {grid_index_1_based} (Parquet): {parquet_filename}")

        # Check if parquet file was actually created (might be skipped if no points)
        if not os.path.exists(parquet_filename):
             logging.warning(f"Parquet file {parquet_filename} not created for grid {grid_index_1_based}. Skipping GeoJSON.")
             return i # Still count as processed, but no GeoJSON

        # 2. Convert Parquet to GeoJSON
        df = pd.read_parquet(parquet_filename)
        if df.empty:
            logging.warning(f"Parquet file {parquet_filename} is empty. Skipping GeoJSON.")
            return i # Count as processed

        # Ensure 'lng' and 'lat' columns exist
        if 'lng' not in df.columns or 'lat' not in df.columns:
             raise ValueError(f"'lng' or 'lat' column not found in {parquet_filename}")

        geometry = [Point(lon, lat) for lon, lat in zip(df['lng'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326") # WGS84
        gdf.to_file(geojson_filename, driver="GeoJSON")
        logging.info(f"Converted grid {grid_index_1_based} to GeoJSON: {geojson_filename}")

        return i # Return index on success

    except Exception as e:
        logging.error(f"Error processing grid {grid_index_1_based} (offset {x_offset},{y_offset}): {e}", exc_info=True)
        # Optionally remove partially created files
        if os.path.exists(parquet_filename): os.remove(parquet_filename)
        if os.path.exists(geojson_filename): os.remove(geojson_filename)
        return None # Indicate failure

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generates multiple K-set sample grids (Parquet and GeoJSON) with unique random offsets.")
    parser.add_argument("--project", type=str, required=True, dest="project_boundary_filename", help="GeoJSON File of project boundary.")
    parser.add_argument("--start_year", type=int, required=True, help="Year project started.")
    parser.add_argument("--evaluation_year", type=int, required=True, help="Year of project evaluation.")
    parser.add_argument("--jrc", type=str, required=True, dest="jrc_directory_path", help="Directory containing JRC AnnualChange GeoTIFF tiles.")
    parser.add_argument("--fcc", type=str, required=True, dest="fcc_directory_path", help="Directory containing FCC GeoTIFF tiles.")
    parser.add_argument("--ecoregions", type=str, required=True, dest="ecoregions_directory_path", help="Directory containing Ecoregions GeoTIFF tiles.")
    parser.add_argument("--elevation", type=str, required=True, dest="elevation_directory_path", help="Directory containing SRTM elevation GeoTIFF tiles.")
    parser.add_argument("--slope", type=str, required=True, dest="slope_directory_path", help="Directory containing slope GeoTIFF tiles.")
    parser.add_argument("--access", type=str, required=True, dest="access_directory_path", help="Directory containing access GeoTIFF tiles.")
    parser.add_argument("--countries-raster", type=str, required=True, dest="countries_raster_filename", help="Raster file of country boundaries.")
    parser.add_argument("--output", type=str, required=True, dest="output_directory", help="Destination directory for k_*.parquet files and geojsons subfolder.")
    parser.add_argument("--buffer", type=int, default=0, required=False, help="Optional: Buffer distance in metres to apply to the project boundary for sampling (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Random number seed for generating offsets.")
    parser.add_argument("--num-grids", type=int, default=250, help="Number of unique random grid offsets to generate.")
    parser.add_argument("--processes", type=int, default=max(1, multiprocessing.cpu_count() // 4), help="Number of parallel processes to use (default: 1/4 of CPU cores).") # Use integer division and ensure at least 1

    args = parser.parse_args()

    # --- Directory Setup ---
    output_directory = args.output_directory
    geojsons_directory = os.path.join(output_directory, "geojsons")
    try:
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(geojsons_directory, exist_ok=True)
        logging.info(f"Output directory: {output_directory}")
        logging.info(f"GeoJSONs directory: {geojsons_directory}")
    except OSError as e:
        logging.error(f"Failed to create output directories: {e}")
        return # Exit if directories can't be created

    # --- Determine Pixel Skip ---
    try:
        project_gdf = gpd.read_file(args.project_boundary_filename)
        project_area_m2 = area_for_geometry(project_gdf) # Ensure this function handles GeoDataFrames
        project_area_ha = project_area_m2 / 10_000
        pixel_skip = PIXEL_SKIP_LARGE_PROJECT if (project_area_ha > 250_000) else PIXEL_SKIP_SMALL_PROJECT
        logging.info(f"Project area: {project_area_ha:.2f} ha. Using pixel_skip: {pixel_skip}")
    except Exception as e:
        logging.error(f"Failed to read project boundary or calculate area: {e}")
        return

    # --- Generate Unique Random Offset Pairs ---
    num_grids_requested = args.num_grids
    random.seed(args.seed)

    # Calculate total possible unique grids
    total_possible_grids = pixel_skip * pixel_skip
    logging.info(f"Total possible unique grid offsets for pixel_skip={pixel_skip}: {total_possible_grids}")

    if num_grids_requested > total_possible_grids:
        logging.warning(f"Requested number of grids ({num_grids_requested}) exceeds the total possible unique grids ({total_possible_grids}). Capping at {total_possible_grids}.")
        num_grids_to_generate = total_possible_grids
    else:
        num_grids_to_generate = num_grids_requested

    # Generate all possible pairs
    all_possible_pairs = list(product(range(pixel_skip), range(pixel_skip)))

    # Sample unique pairs
    selected_pairs = random.sample(all_possible_pairs, num_grids_to_generate)

    # Unzip into separate lists
    x_offsets, y_offsets = zip(*selected_pairs)
    # Convert tuples from zip to lists as expected by worker
    x_offsets = list(x_offsets)
    y_offsets = list(y_offsets)

    logging.info(f"Generated {num_grids_to_generate} unique offset pairs with seed {args.seed}.")

    # --- Prepare for Parallel Processing ---
    worker_func = partial(
        process_single_grid_task,
        args=args,
        pixel_skip=pixel_skip,
        x_offsets=x_offsets, # Pass the list of selected x offsets
        y_offsets=y_offsets, # Pass the list of selected y offsets
        output_directory=output_directory,
        geojsons_directory=geojsons_directory
    )

    # --- Run Parallel Processing ---
    num_processes = min(args.processes, num_grids_to_generate) # Don't use more processes than tasks
    logging.info(f"Starting parallel generation of {num_grids_to_generate} grids using {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker_func, range(num_grids_to_generate)) # Iterate up to the number generated

    # --- Summarize Results ---
    successful_grids = sum(1 for r in results if r is not None)
    failed_grids = num_grids_to_generate - successful_grids
    logging.info(f"Finished generation.")
    logging.info(f"Successfully processed: {successful_grids}/{num_grids_to_generate} grids.")
    if failed_grids > 0:
        logging.warning(f"Failed to process: {failed_grids}/{num_grids_to_generate} grids. Check logs for errors.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()