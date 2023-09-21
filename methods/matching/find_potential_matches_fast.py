import argparse
from collections import defaultdict
import glob
import logging
import math
import os
import sys
import time
from multiprocessing import Manager, Process, Queue, cpu_count

from osgeo import gdal  # type: ignore
import numpy as np
import pandas as pd
from methods.utils.dranged_tree import DRangedTree
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection
from methods.common.luc import luc_matching_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# We do not re-use data in this, so set a small block cache size for GDAL, otherwise
# it pointlessly hogs memory, and then spends a long time tidying it up after.
gdal.SetCacheMax(1024 * 1024 * 16)

def build_key(ecoregion, country, luc0, luc5, luc10):
    return  (int(ecoregion) << 32) | (int(country) << 16) | (int(luc0) << 10) | (int(luc5) << 5) | (int(luc10))

def key_builder(start_year: int):
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    def _build_key(row):
        return  build_key(row.ecoregion, row.country, row[luc0], row[luc5], row[luc10])
    return _build_key

from pprint import pprint

def debug(msg, item):
    print('\n' + msg + ':')
    pprint(item)
    return item

def load_k(
    k_filename: str,
    start_year: int,
    worker_count: int,
    ktree_queue: Queue,
) -> None:
    print("Reading k...")
    source_pixels = pd.read_parquet(k_filename)
    
    # Split source_pixels into classes
    source_classes = defaultdict(list)
    build_key = key_builder(start_year)

    for _, row in source_pixels.iterrows():
        key = build_key(row)
        source_classes[key].append(row)

    print("Building k trees...")

    source_trees = dict()
    for key, values in source_classes.items():
        source_trees[key] = DRangedTree.build(np.array([(
            row.elevation,
            row.slope,
            row.access,
            row["cpc0_u"],
            row["cpc0_d"],
            row["cpc5_u"],
            row["cpc5_d"],
            row["cpc10_u"],
            row["cpc10_d"],
            ) for row in values
        ]), np.array([
            200,
            2.5,
            10,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
        ]))

    print("Sending k trees to workers...")

    for _ in range(worker_count):
        ktree_queue.put(source_trees)


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
    ktree_queue: Queue,
    coordinate_queue: Queue,
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    ktrees = ktree_queue.get()

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

    result_path = os.path.join(result_folder, f"fast_{worker_index}.tif")

    matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=result_path)
    xsize = matching_collection.boundary.window.xsize
    ysize = matching_collection.boundary.window.ysize
    print("Size", xsize, ysize)
    xstride = math.ceil(xsize / 10)
    ystride = math.ceil(ysize / 10)

    # Iterate our assigned pixels
    while True:
        coords = coordinate_queue.get()
        print("Processing coordinates: ", coords)
        if coords == None:
            # Finished
            return
        y, x = coords
        ymin = y * ystride
        xmin = x * xstride
        ymax = min(ymin + ystride, ysize)
        xmax = min(xmin + xstride, xsize)
        xwidth = xmax - xmin
        ywidth = ymax - ymin
        print(f"Reading {xmin}, {ymin} width {xwidth} height {ywidth}...")
        elevations = matching_collection.elevation.read_array(xmin, ymin, xwidth, ywidth)
        ecoregions = matching_collection.ecoregions.read_array(xmin, ymin, xwidth, ywidth)
        slopes = matching_collection.slope.read_array(xmin, ymin, xwidth, ywidth)
        accesses = matching_collection.access.read_array(xmin, ymin, xwidth, ywidth)
        lucs = [x.read_array(xmin, ymin, xwidth, ywidth) for x in matching_collection.lucs]

        for cpc in matching_collection.cpcs:
            print("CPC size", cpc.window.xsize, cpc.window.ysize)
        cpcs = [debug(f"read_array {index}", x.read_array(xmin, ymin, xwidth, ywidth)) for index, x in enumerate(matching_collection.cpcs)]

        countries = matching_collection.countries.read_array(xmin, ymin, xwidth, ywidth)
        points = np.zeros((ywidth, xwidth))
        print("Processing block")
        for y in range(ywidth):
            if y%10 == 0:
                print(f"y: {y} of {ywidth}")
            for x in range(xwidth):
                # FIXME: Check in boundary
                ecoregion = ecoregions[y, x]
                country = countries[y, x]
                luc0 = lucs[0][y, x]
                luc5 = lucs[1][y, x]
                luc10 = lucs[2][y, x]
                key = build_key(ecoregion, country, luc0, luc5, luc10)
                if key in ktrees:
                    points[y, x] = 1 if ktrees[key].contains(np.array([
                        elevations[y, x],
                        slopes[y, x],
                        accesses[y, x],
                        cpcs[0][y, x],
                        cpcs[1][y, x],
                        cpcs[2][y, x],
                        cpcs[3][y, x],
                        cpcs[4][y, x],
                        cpcs[5][y, x],
                    ])) else 0
        
        # Write points to output
        matching_pixels._dataset.GetRasterBand(1).WriteArray(points, xmin, ymin)


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

    with Manager() as manager:
        ktree_queue = manager.Queue()
        coordinate_queue = manager.Queue()

        worker_count = processes_count

        # Fill the co-ordinate queue
        #for y in range(10):
        #    for x in range(10):
        #        coordinate_queue.put([y, x])
        coordinate_queue.put([6, 7])
        for _ in range(worker_count):
            coordinate_queue.put(None)

        # We allocate an extra process for this since whilst it is running, the workers are idle
        # and once it sends anything to the works, then it stops.
        ktree_builder_process = Process(target=load_k, args=(k_filename, start_year, worker_count, ktree_queue))
        ktree_builder_process.start()

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
            ktree_queue,
            coordinate_queue,
        )) for index in range(worker_count)]
        for worker_process in workers:
            worker_process.start()

        processes = workers + [ktree_builder_process]
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
