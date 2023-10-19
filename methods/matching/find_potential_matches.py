import argparse
import glob
import logging
import os
import sys
import time
from multiprocessing import Manager, Process, Queue, cpu_count

from osgeo import gdal  # type: ignore
import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection
from methods.common.luc import luc_matching_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# We do not re-use data in this, so set a small block cache size for GDAL, otherwise
# it pointlessly hogs memory, and then spends a long time tidying it up after.
gdal.SetCacheMax(1024 * 1024 * 16)

def load_k(
    k_filename: str,
    sentinal_count: int,
    output_queue: Queue,
) -> None:
    # put the source pixels into the queue
    source_pixels = pd.read_parquet(k_filename)
    for row in source_pixels.iterrows():
        output_queue.put(row)
    # To close the pipe put in one sentinel value per worker
    for _ in range(sentinal_count):
        output_queue.put(None)

def worker(
    worker_index: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    start_year: int,
    _evaluation_year: int,
    result_folder: str,
    input_queue: Queue
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    luc0, luc5, luc10 = luc_matching_columns(start_year)

    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        [start_year, start_year - 5, start_year - 10],
        [start_year, start_year - 5, start_year - 10],
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
        countries_raster_filename,
    )

    # To help with such a long running job, we log every 100th element
    # motivated by issues with the pipeline hanging before during multi hour jobs
    count = 0
    while True:
        if count % 100 == 0:
            logging.info("%d: processing %d", worker_index, count)
        count += 1

        row = input_queue.get()
        if row is None:
            break
        index, matching = row

        result_path = os.path.join(result_folder, f"{index}.tif")
        if os.path.exists(result_path):
            try:
                raster = RasterLayer.layer_from_file(result_path)
                if raster.sum() > 0:
                    continue
            except FileNotFoundError:
                pass

        matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=result_path)

        filtered_ecoregions = matching_collection.ecoregions.numpy_apply(lambda chunk: chunk == matching.ecoregion)
        filtered_elevation = matching_collection.elevation.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.elevation - 200), chunk <= (matching.elevation + 200))
        )
        filtered_slopes = matching_collection.slope.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.slope - 2.5), chunk <= (matching.slope + 2.5))
        )
        filtered_access = matching_collection.access.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.access - 10), chunk <= (matching.access + 10))
        )
        filtered_luc0 = matching_collection.lucs[0].numpy_apply(lambda chunk: chunk == matching[luc0])
        filtered_luc5 = matching_collection.lucs[1].numpy_apply(lambda chunk: chunk == matching[luc5])
        filtered_luc10 = matching_collection.lucs[2].numpy_apply(lambda chunk: chunk == matching[luc10])

        # +/- 10% of CPC_U and CPC_D for t0, t-5 and t-10
        filtered_cpc_u0 = matching_collection.cpcs[0].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc0_u"] - matching["cpc0_u"] * 0.1),
                chunk <= (matching["cpc0_u"] + matching["cpc0_u"] * 0.1)
            )
        )
        filtered_cpc_d0 = matching_collection.cpcs[1].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc0_d"] - matching["cpc0_d"] * 0.1),
                chunk <= (matching["cpc0_d"] + matching["cpc0_d"] * 0.1)
            )
        )
        filtered_cpc_u5 = matching_collection.cpcs[2].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc5_u"] - matching["cpc5_u"] * 0.1),
                chunk <= (matching["cpc5_u"] + matching["cpc5_u"] * 0.1)
            )
        )
        filtered_cpc_d5 = matching_collection.cpcs[3].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc5_d"] - matching["cpc5_d"] * 0.1),
                chunk <= (matching["cpc5_d"] + matching["cpc5_d"] * 0.1)
            )
        )
        filtered_cpc_u10 = matching_collection.cpcs[4].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc10_u"] - matching["cpc10_u"] * 0.1),
                chunk <= (matching["cpc10_u"] + matching["cpc10_u"] * 0.1)
            )
        )
        filtered_cpc_d10 = matching_collection.cpcs[5].numpy_apply(
            lambda chunk: np.logical_and(
                chunk >= (matching["cpc10_d"] - matching["cpc10_d"] * 0.1),
                chunk <= (matching["cpc10_d"] + matching["cpc10_d"] * 0.1)
            )
        )

        filtered_countries = matching_collection.countries.numpy_apply(lambda chunk: chunk == matching.country)

        calc = matching_collection.boundary * filtered_ecoregions * filtered_elevation * filtered_countries * \
            filtered_luc0 * filtered_luc5 * filtered_luc10 * filtered_slopes * filtered_access * filtered_cpc_u0 * \
            filtered_cpc_u5 * filtered_cpc_u10 * filtered_cpc_d0 * filtered_cpc_d5 * filtered_cpc_d10
        calc.save(matching_pixels)


def find_potential_matches(
    k_filename: str,
    start_year: int,
    evaluation_year: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    result_folder: str,
    processes_count: int
) -> None:
    os.makedirs(result_folder, exist_ok=True)

    assert processes_count >= 3

    with Manager() as manager:
        source_queue = manager.Queue()

        worker_count = processes_count - 2
        workers = [Process(target=worker, args=(
            index,
            matching_zone_filename,
            jrc_directory_path,
            cpc_directory_path,
            ecoregions_directory_path,
            elevation_directory_path,
            slope_directory_path,
            access_directory_path,
            countries_raster_filename,
            start_year,
            evaluation_year,
            result_folder,
            source_queue
        )) for index in range(worker_count)]
        for worker_process in workers:
            worker_process.start()
        ingest_process = Process(target=load_k, args=(k_filename, worker_count, source_queue))
        ingest_process.start()

        processes = workers + [ingest_process]
        while processes:
            candidates = [x for x in processes if not x.is_alive()]
            for candidate in candidates:
                candidate.join()
                if candidate.exitcode:
                    for victim in processes:
                        victim.kill()
                    sys.exit(candidate.exitcode)
                processes.remove(candidate)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Generates a set of rasters per entry in K with potential matches.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_filename",
        help="Parquet file containing pixels from K as generated by calculate_k.py"
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
        "--cpc",
        type=str,
        required=True,
        dest="cpc_directory_path",
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
        help="Destination directory for storing per-K rasters."
    )
    parser.add_argument(
        "--countries-raster",
        type=str,
        required=True,
        dest="countries_raster_filename",
        help="Raster of country IDs."
    )
    parser.add_argument(
        "-j",
        type=int,
        required=False,
        default=round(cpu_count() / 2),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    find_potential_matches(
        args.k_filename,
        args.start_year,
        args.evaluation_year,
        args.matching_zone_filename,
        args.jrc_directory_path,
        args.cpc_directory_path,
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
