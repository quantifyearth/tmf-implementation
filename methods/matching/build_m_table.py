import argparse
import glob
import os
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
import re  # Import regex module
import numpy as np
import tempfile

import polars as pl
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection
from methods.common.luc import luc_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(processName)s %(message)s")

# --- process_partial_raster function ---
def process_partial_raster(
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
    temp_output_dir: str,
    strip_index: int,
    partial_raster_path: str
) -> str | None:
    """Processes one partial M raster row by row and saves its table data to a temp parquet file."""
    basename = os.path.basename(partial_raster_path)
    temp_parquet_filename = os.path.join(temp_output_dir, f"temp_m_table_{strip_index:04d}.parquet")
    logging.debug(f"Processing {basename} (strip {strip_index}) -> {temp_parquet_filename}")

    try:
        partial_raster = RasterLayer.layer_from_file(partial_raster_path)
        with partial_raster:
            # Define years needed for FCC relative naming *before* building collection
            fcc_years_absolute = [start_year, start_year - 5, start_year - 10]
            fcc_years_relative = [0, 5, 10] # For naming columns

            # Build the collection - needed to access the layers
            matching_collection = build_layer_collection(
                partial_raster.pixel_scale,
                partial_raster.projection,
                list(luc_range(start_year, evaluation_year)),
                fcc_years_absolute, # Pass absolute years to build_layer_collection
                matching_zone_filename,
                jrc_directory_path,
                fcc_directory_path,
                ecoregions_directory_path,
                elevation_directory_path,
                slope_directory_path,
                access_directory_path,
                countries_raster_filename,
            )

            results = [] # List to store data for matching pixels
            luc_years = list(luc_range(start_year, evaluation_year))
            fcc_suffixes = ['u', 'd']

            window = partial_raster.window
            width = window.xsize
            logging.debug(f"Processing window {window} for {basename} row by row")

            # Iterate through rows
            for yoffset in range(window.ysize):
                # Read the current row from the M raster
                row_matches = partial_raster.read_array(0, yoffset, width, 1)

                # Check if any matches exist in this row before reading other layers
                if not row_matches.any():
                    continue

                # Read the corresponding row from all other layers
                row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
                row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
                row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
                row_access = matching_collection.access.read_array(0, yoffset, width, 1)
                row_country = matching_collection.countries.read_array(0, yoffset, width, 1)

                rows_luc = {}
                for year, layer in zip(luc_years, matching_collection.lucs):
                    rows_luc[year] = layer.read_array(0, yoffset, width, 1)

                rows_fcc = {}
                fcc_idx = 0
                # Iterate using absolute years to match the layers read by build_layer_collection
                for year_abs in fcc_years_absolute:
                    for suffix in fcc_suffixes:
                        if fcc_idx < len(matching_collection.fccs):
                            # Store the row data using a tuple key (absolute_year, suffix)
                            rows_fcc[(year_abs, suffix)] = matching_collection.fccs[fcc_idx].read_array(0, yoffset, width, 1)
                        else:
                            # Create a row of NaNs if layer is missing
                            rows_fcc[(year_abs, suffix)] = np.full((1, width), np.nan, dtype=np.float32)
                        fcc_idx += 1

                # Iterate through columns in the row
                for xoffset in range(width):
                    if row_matches[0, xoffset] > 0: # Check if this pixel is a match
                        # Get coordinates for the matching pixel
                        lat, lng = partial_raster.latlng_for_pixel(xoffset, yoffset)

                        # Extract data using NumPy indexing
                        pixel_data = {
                            'lat': lat,
                            'lng': lng,
                            'strip_id': strip_index,
                            'ecoregion': row_ecoregion[0, xoffset],
                            'elevation': row_elevation[0, xoffset],
                            'slope': row_slope[0, xoffset],
                            'access': row_access[0, xoffset],
                            'country': row_country[0, xoffset],
                        }

                        # Add LUC data
                        for year in luc_years:
                            pixel_data[f'luc_{year}'] = rows_luc[year][0, xoffset]

                        # Add FCC data using relative year naming (fcc0_u, fcc5_d, etc.)
                        for year_rel, year_abs in zip(fcc_years_relative, fcc_years_absolute):
                            for suffix in fcc_suffixes:
                                # Use the relative year for the key name
                                key_name = f'fcc{year_rel}_{suffix}'
                                # Access the row data using the absolute year and suffix
                                pixel_data[key_name] = rows_fcc[(year_abs, suffix)][0, xoffset]

                        results.append(pixel_data)

            # After processing all rows, check if any results were found
            if not results:
                logging.debug(f"No matching pixels found in {basename} after full scan. Skipping.")
                return None

            # Create Polars DataFrame from the list of dictionaries
            output_df = pl.DataFrame(results)

            # Define column order using relative FCC names
            luc_columns = [f'luc_{year}' for year in luc_years]
            # Use relative years for FCC column names
            fcc_columns = [f'fcc{year_rel}_{suffix}' for year_rel in fcc_years_relative for suffix in fcc_suffixes]
            final_columns = ['lat', 'lng', 'strip_id', 'ecoregion', 'elevation', 'slope', 'access', 'country'] + luc_columns + fcc_columns
            output_df = output_df.select(final_columns) # Reorder columns

            output_df.write_parquet(temp_parquet_filename)
            logging.debug(f"Wrote {len(output_df)} points from {basename} to {temp_parquet_filename}")
            return temp_parquet_filename

    except Exception as e:
        logging.error(f"Error processing {basename}: {e}", exc_info=True)
        return None

# --- Helper function to extract strip index (unchanged) ---
def extract_strip_index(filename: str) -> int:
    match = re.search(r'strip_(\d+)\.tif$', os.path.basename(filename))
    if match:
        return int(match.group(1))
    logging.warning(f"Could not extract strip index from filename: {filename}. Assigning high index.")
    return float('inf')

# --- main function (largely unchanged, ensure imports are correct) ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Finds all potential matches to K in matching zone, aka set M.")
    parser.add_argument(
        "--rasters_directory",
        type=str,
        required=True,
        dest="m_rasters_directory",
        help="Directory containing partial M rasters (*.tif) generated by find_potential_matches.py"
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
        help="Year of project evaluation"
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
        help="Raster of country IDs."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination parquet file for results."
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

    glob_pattern = os.path.join(args.m_rasters_directory, "strip_*.tif")
    partial_raster_files = glob.glob(glob_pattern)
    if not partial_raster_files:
        logging.error(f"No strip_*.tif files found in {args.m_rasters_directory}")
        return

    partial_raster_files.sort(key=extract_strip_index)
    logging.info(f"Found and sorted {len(partial_raster_files)} partial M rasters to process.")

    tasks = []
    for file_path in partial_raster_files:
        index = extract_strip_index(file_path)
        if index != float('inf'):
            tasks.append((index, file_path))
        else:
            logging.warning(f"Skipping file due to naming pattern mismatch: {file_path}")

    if not tasks:
        logging.error("No valid strip files found to process after filtering.")
        return

    with tempfile.TemporaryDirectory() as tempdir:
        logging.info(f"Using temporary directory for intermediate parquets: {tempdir}")

        worker_func = partial(
            process_partial_raster,
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
            tempdir
        )

        num_processes = min(args.processes_count, len(tasks))
        logging.info(f"Starting parallel processing with {num_processes} workers...")
        temp_parquet_files = []
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(worker_func, tasks)
            temp_parquet_files = [f for f in results if f is not None and os.path.exists(f)]

        logging.info(f"Finished processing individual strips. {len(temp_parquet_files)} temporary files generated.")

        if temp_parquet_files:
            logging.info(f"Concatenating {len(temp_parquet_files)} intermediate parquet files...")
            combined_df = pl.scan_parquet(temp_parquet_files).collect(streaming=True)

            logging.info("Sorting combined data by strip_id, lat, lng...")
            combined_df = combined_df.sort(['strip_id', 'lat', 'lng'])

            logging.info(f"Writing final sorted table with {len(combined_df)} rows to {args.output_filename}")
            combined_df.write_parquet(args.output_filename)
            logging.info(f"Output successfully written to {args.output_filename}")
        else:
            logging.warning("No valid intermediate parquet files were generated. Output file will not be created.")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
