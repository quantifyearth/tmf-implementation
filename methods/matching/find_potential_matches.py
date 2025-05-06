import argparse
from collections import defaultdict
import glob
import logging
import math
import os
import sys
import time
from multiprocessing import Manager, Process, Queue, cpu_count
from typing import Mapping
from osgeo import gdal, gdal_array  # type: ignore # Import gdal_array
import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.common.luc import luc_matching_columns
from methods.matching.calculate_k import build_layer_collection
from methods.utils.dranged_tree import DRangedTree

DIVISIONS = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# We do not re-use data in this, so set a small block cache size for GDAL, otherwise
# it pointlessly hogs memory, and then spends a long time tidying it up after.
gdal.SetCacheMax(1024 * 1024 * 16)

def build_key(ecoregion, country, luc0, luc5, luc10):
    """Create a 64-bit key for fields that must match exactly"""
    if ecoregion < 0 or ecoregion > 0x7fffffff:
        raise ValueError("Ecoregion doesn't fit in 31 bits")
    if country < 0 or country > 0xffff:
        raise ValueError("Country doesn't fit in 16 bits")
    if luc0 < 0 or luc0 > 0x1f:
        raise ValueError("luc0 doesn't fit in 5 bits")
    if luc5 < 0 or luc5 > 0x1f:
        raise ValueError("luc5 doesn't fit in 5 bits")
    if luc10 < 0 or luc10 > 0x1f:
        raise ValueError("luc10 doesn't fit in 5 bits")
    return  (int(ecoregion) << 32) | (int(country) << 16) | (int(luc0) << 10) | (int(luc5) << 5) | (int(luc10))

def key_builder(start_year: int):
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    def _build_key(row):
        return  build_key(row.ecoregion, row.country, row[luc0], row[luc5], row[luc10])
    return _build_key

def load_k(
    k_directory: str,
    start_year: int,
) -> Mapping[int, DRangedTree]:

    logging.info(f"Reading K set Parquet files from directory: {k_directory}")
    parquet_files = glob.glob(os.path.join(k_directory, "k_*.parquet"))

    if not parquet_files:
        logging.error(f"No Parquet files found in directory: {k_directory}")
        sys.exit(1)

    logging.info(f"Found {len(parquet_files)} Parquet files. Concatenating...")

    # Read and concatenate all found parquet files
    df_list = [pd.read_parquet(f) for f in parquet_files]
    source_pixels = pd.concat(df_list, ignore_index=True)

    logging.info(f"Concatenated K set contains {len(source_pixels)} total points.")

    # Split source_pixels into classes
    source_classes = defaultdict(list)
    build_key_for_row = key_builder(start_year)

    for _, row in source_pixels.iterrows():
        key = build_key_for_row(row)
        source_classes[key].append(row)

    logging.info("Building k trees...")

    source_trees = {}
    for key, values in source_classes.items():
        source_trees[key] = DRangedTree.build(
            np.array([(
                row.elevation,
                row.slope,
                row.access,
                row["fcc0_u"],
                row["fcc0_d"],
                row["fcc5_u"],
                row["fcc5_d"],
                row["fcc10_u"],
                row["fcc10_d"],
                ) for row in values
            ]),
            np.array([
                200,
                2.5,
                10,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ]),
            1 / 100, # This is the fraction of R that is in M, used to optimize search speed.
        )

    logging.info(f"Built {len(source_trees)} k trees.")

    return source_trees

def worker(
    worker_index: int, # Keep worker_index for logging if needed
    matching_zone_filename: str,
    jrc_directory_path: str,
    fcc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    start_year: int,
    _evaluation_year: int,
    result_folder: str,
    ktrees: Mapping[int, DRangedTree],
    coordinate_queue: Queue,
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    # Build the layer collection once per worker
    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        [start_year, start_year - 5, start_year - 10],
        [start_year, start_year - 5, start_year - 10],
        matching_zone_filename,
        jrc_directory_path,
        fcc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
        countries_raster_filename,
    )

    # Get overall dimensions and stride
    xsize = matching_collection.boundary.window.xsize
    ysize = matching_collection.boundary.window.ysize
    xstride = math.ceil(xsize) # Process full width
    ystride = math.ceil(ysize / DIVISIONS)

    # Iterate our assigned pixels
    while True:
        coords = coordinate_queue.get()
        if coords is None:
            logging.debug(f"Worker {worker_index} received None, finishing.")
            break

        ypos, xpos = coords # xpos should always be 0 here as we process full width
        logging.debug(f"Worker {worker_index} starting strip ypos={ypos}...")

        # --- Define output path based on the strip (ypos) ---
        result_path = os.path.join(result_folder, f"strip_{ypos:04d}.tif") # Use ypos for consistent naming

        # --- Create the output layer for this specific strip ---
        # Use the overall layer's properties but specify the filename for this strip
        try:
            matching_pixels = RasterLayer.empty_raster_layer_like(
                matching_collection.boundary,
                filename=result_path
            )
        except Exception as e:
            logging.error(f"Worker {worker_index} failed to create output file {result_path}: {e}")
            continue # Skip this strip if file creation fails

        # Calculate bounds for this strip
        ymin = ypos * ystride
        xmin = 0 # Start at the left edge
        ymax = min(ymin + ystride, ysize)
        xmax = xsize # Go to the right edge
        xwidth = xmax - xmin
        ywidth = ymax - ymin

        if xwidth <= 0 or ywidth <= 0:
            logging.warning(f"Worker {worker_index} strip ypos={ypos} resulted in zero dimensions, skipping.")
            # Clean up the potentially created empty file
            try:
                del matching_pixels._dataset # Release handle if possible
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception as cleanup_e:
                 logging.error(f"Worker {worker_index} error cleaning up {result_path}: {cleanup_e}")
            continue

        # Read data for the strip
        try:
            boundary = matching_collection.boundary.read_array(xmin, ymin, xwidth, ywidth)
            elevations = matching_collection.elevation.read_array(xmin, ymin, xwidth, ywidth)
            ecoregions = matching_collection.ecoregions.read_array(xmin, ymin, xwidth, ywidth)
            slopes = matching_collection.slope.read_array(xmin, ymin, xwidth, ywidth)
            accesses = matching_collection.access.read_array(xmin, ymin, xwidth, ywidth)
            lucs = [x.read_array(xmin, ymin, xwidth, ywidth) for x in matching_collection.lucs]
            fccs = [fcc.read_array(xmin, ymin, xwidth, ywidth) for fcc in matching_collection.fccs]
            countries = matching_collection.countries.read_array(xmin, ymin, xwidth, ywidth)
        except Exception as read_e:
            logging.error(f"Worker {worker_index} failed reading data for strip ypos={ypos}: {read_e}")
            # Clean up the potentially created empty file
            try:
                del matching_pixels._dataset # Release handle if possible
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception as cleanup_e:
                 logging.error(f"Worker {worker_index} error cleaning up {result_path}: {cleanup_e}")
            continue # Skip processing this strip

        # Process pixels within the strip
        try:
            # Convert GDAL datatype code to NumPy dtype
            numpy_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(matching_pixels.datatype)
            points = np.zeros((ywidth, xwidth), dtype=numpy_dtype) # Use the converted NumPy dtype
        except Exception as dtype_e:
             logging.error(f"Worker {worker_index} failed to determine numpy dtype from GDAL type {matching_pixels.datatype} for strip ypos={ypos}: {dtype_e}")
             # Clean up the potentially created empty file
             try:
                 del matching_pixels._dataset # Release handle if possible
                 if os.path.exists(result_path):
                     os.remove(result_path)
             except Exception as cleanup_e:
                 logging.error(f"Worker {worker_index} error cleaning up {result_path} after dtype failure: {cleanup_e}")
             continue # Skip this strip

        for y_local in range(ywidth):
            for x_local in range(xwidth):
                if boundary[y_local, x_local] == 0:
                    continue
                ecoregion = ecoregions[y_local, x_local]
                country = countries[y_local, x_local]
                luc0 = lucs[0][y_local, x_local]
                luc5 = lucs[1][y_local, x_local]
                luc10 = lucs[2][y_local, x_local]

                # Check for nodata or invalid values before building key
                if any(val is None or val < 0 for val in [ecoregion, country, luc0, luc5, luc10]):
                    continue

                try:
                    key = build_key(ecoregion, country, luc0, luc5, luc10)
                except ValueError: # Catch errors from build_key if values are out of expected range
                    continue

                if key in ktrees:
                    # Check for nodata in continuous variables
                    covariates = np.array([
                        elevations[y_local, x_local],
                        slopes[y_local, x_local],
                        accesses[y_local, x_local],
                        fccs[0][y_local, x_local], # fcc0_u
                        fccs[1][y_local, x_local], # fcc0_d
                        fccs[2][y_local, x_local], # fcc5_u
                        fccs[3][y_local, x_local], # fcc5_d
                        fccs[4][y_local, x_local], # fcc10_u
                        fccs[5][y_local, x_local], # fcc10_d
                    ])
                    # Assuming nodata is represented by NaN or a specific negative value; adjust check if needed
                    if np.isnan(covariates).any() or (covariates < -999).any(): # Example check
                        continue

                    try:
                        if ktrees[key].contains(covariates):
                            points[y_local, x_local] = 1
                    except Exception as tree_e:
                         logging.warning(f"Worker {worker_index} error during tree check for strip ypos={ypos}, pixel ({x_local},{y_local}): {tree_e}")
                         continue # Skip this pixel if tree check fails

        # Write points to the output file for this strip
        try:
            # pylint: disable-next=protected-access
            matching_pixels._dataset.GetRasterBand(1).WriteArray(points, xmin, ymin)
            # Explicitly close the dataset for this strip to ensure it's written
            del matching_pixels._dataset
            logging.debug(f"Worker {worker_index} completed and saved strip ypos={ypos} to {result_path}.")
        except Exception as write_e:
            logging.error(f"Worker {worker_index} failed writing data for strip ypos={ypos} to {result_path}: {write_e}")
            # Attempt cleanup if write fails
            try:
                if '_dataset' in locals() and matching_pixels._dataset is not None:
                     del matching_pixels._dataset
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception as cleanup_e:
                 logging.error(f"Worker {worker_index} error cleaning up {result_path} after write failure: {cleanup_e}")

    logging.info(f"Worker {worker_index} finished.")
    # Removed the final del matching_pixels._dataset as it's handled per strip


def find_potential_matches(
    k_directory: str,
    start_year: int,
    evaluation_year: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    fcc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    result_folder: str,
    processes_count: int
) -> None:
    os.makedirs(result_folder, exist_ok=True)

    with Manager() as manager:
        coordinate_queue = manager.Queue()

        worker_count = processes_count

        # Fill the co-ordinate queue with strip indices (ypos)
        for ypos in range(DIVISIONS):
            coordinate_queue.put([ypos, 0]) # Keep format [ypos, xpos], xpos is always 0
        # Add termination signals for workers
        for _ in range(worker_count):
            coordinate_queue.put(None)

        ktree = load_k(k_directory, start_year)

        workers = [Process(target=worker, args=(
            index,
            matching_zone_filename,
            jrc_directory_path,
            fcc_directory_path,
            ecoregions_directory_path,
            elevation_directory_path,
            slope_directory_path,
            access_directory_path,
            countries_raster_filename,
            start_year,
            evaluation_year,
            result_folder,
            ktree,
            coordinate_queue,
        )) for index in range(worker_count)]
        for worker_process in workers:
            worker_process.start()

        # Monitor worker processes
        active_workers = list(workers) # Create a copy to modify
        while active_workers:
            finished_workers = []
            for proc in active_workers:
                if not proc.is_alive():
                    proc.join()
                    if proc.exitcode != 0:
                        logging.error(f"Worker process {proc.pid} exited with code {proc.exitcode}. Terminating others.")
                        # Terminate remaining workers
                        for other_proc in active_workers:
                            if other_proc.is_alive():
                                other_proc.terminate()
                                other_proc.join(timeout=5) # Wait briefly for termination
                                if other_proc.is_alive():
                                     other_proc.kill() # Force kill if terminate fails
                                     other_proc.join()
                        sys.exit(f"Worker process failed with exit code {proc.exitcode}")
                    finished_workers.append(proc)

            # Remove finished workers from the active list
            for proc in finished_workers:
                active_workers.remove(proc)

            if active_workers: # Avoid busy-waiting if workers are still running
                time.sleep(1)

        logging.info("All worker processes finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Generates a set of rasters per process with potential matches.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_directory",
        help="Directory containing K set Parquet files (k_*.parquet) as generated by calculate_k_fast.py"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file desribing area from which matching pixels may be selected."
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
        "--fcc",
        type=str,
        required=True,
        dest="fcc_directory_path",
        help="Filder containing Coarsened Proportional Coverage GeoTIFF tiles for all years."
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
        "--output",
        type=str,
        required=True,
        dest="output_directory",
        help="Destination directory for storing per-process rasters (horizontally striped)."
    )
    parser.add_argument(
        "--countries-raster",
        type=str,
        required=True,
        dest="countries_raster_filename",
        help="Raster of country IDs."
    )
    parser.add_argument(
        "--j",
        type=int,
        required=False,
        default=round(cpu_count() / 4),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    find_potential_matches(
        args.k_directory,
        args.start_year,
        args.evaluation_year,
        args.matching_zone_filename,
        args.jrc_directory_path,
        args.fcc_directory_path,
        args.ecoregions_directory_path,
        args.elevation_directory_path,
        args.slope_directory_path,
        args.access_directory_path,
        args.countries_raster_filename,
        args.output_directory,
        args.processes_count
    )

if __name__ == "__main__":
    main()