import argparse
from collections import defaultdict
import glob
import logging
import math
import os
import sys
import time
from multiprocessing import Manager, Process, Queue, cpu_count
from typing import Mapping, Tuple
from osgeo import gdal  # type: ignore
import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.common.luc import luc_matching_columns
from methods.matching.calculate_k import build_layer_collection
from methods.utils.dranged_tree import DRangedTree

DIVISIONS = 1000

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
    k_filename: str,
    start_year: int,
) -> Mapping[int, DRangedTree]:

    print("Reading k...")
    source_pixels = pd.read_parquet(k_filename)

    # Split source_pixels into classes
    source_classes = defaultdict(list)
    build_key_for_row = key_builder(start_year)

    for _, row in source_pixels.iterrows():
        key = build_key_for_row(row)
        source_classes[key].append(row)

    print("Building k trees...")

    source_trees = {}
    for key, values in source_classes.items():
        source_trees[key] = DRangedTree.build(
            np.array([(
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

    print("k trees built.")

    return source_trees

def exact_pixel_for_lat_lng(layer, lat: float, lng: float) -> Tuple[int,int]:
    """Get pixel for geo coords. This is relative to the set view window.
    Result is rounded down to nearest pixel."""
    if "WGS 84" not in layer.projection:
        raise NotImplementedError("Not yet supported for other projections")
    pixel_scale = layer.pixel_scale
    if pixel_scale is None:
        raise ValueError("Layer has no pixel scale")
    return (
        (lng - layer.area.left) / pixel_scale.xstep,
        (lat - layer.area.top) / pixel_scale.ystep,
    )

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
    ktrees: Mapping[int, DRangedTree],
    coordinate_queue: Queue,
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

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
    xstride = math.ceil(xsize)
    ystride = math.ceil(ysize / DIVISIONS)

    # Iterate our assigned pixels
    while True:
        coords = coordinate_queue.get()
        if coords is None:
            break
        print(f"Worker {worker_index} starting coords {coords}...")
        ypos, xpos = coords
        ymin = ypos * ystride
        xmin = xpos * xstride
        ymax = min(ymin + ystride, ysize)
        xmax = min(xmin + xstride, xsize)
        xwidth = xmax - xmin
        ywidth = ymax - ymin
        if xwidth <= 0 or ywidth <= 0:
            print(f"Worker {worker_index} coords {coords} are outside boundary")
            continue
        boundary = matching_collection.boundary.read_array(xmin, ymin, xwidth, ywidth)
        elevations = matching_collection.elevation.read_array(xmin, ymin, xwidth, ywidth)
        ecoregions = matching_collection.ecoregions.read_array(xmin, ymin, xwidth, ywidth)
        slopes = matching_collection.slope.read_array(xmin, ymin, xwidth, ywidth)
        accesses = matching_collection.access.read_array(xmin, ymin, xwidth, ywidth)
        lucs = [x.read_array(xmin, ymin, xwidth, ywidth) for x in matching_collection.lucs]

        # FIXME: This still doesn't match perfectly with Patrick's scaled CPCs.
        boundary_tl = matching_collection.boundary.latlng_for_pixel(xmin, ymin)
        cpc_tl = matching_collection.cpcs[0].pixel_for_latlng(*boundary_tl)
        boundary_br = matching_collection.boundary.latlng_for_pixel(xmax, ymax)
        cpc_br = matching_collection.cpcs[0].pixel_for_latlng(*boundary_br)
        cpc_width = cpc_br[0] - cpc_tl[0] + 2 # Get a few spare pixels
        cpc_height = cpc_br[1] - cpc_tl[1] + 2
        cpcs = [
            cpc.read_array(cpc_tl[0], cpc_tl[1], cpc_width, cpc_height)
            for cpc in matching_collection.cpcs
        ]

        exact_cpc_tl = exact_pixel_for_lat_lng(matching_collection.cpcs[0], *boundary_tl)
        exact_cpc_br = exact_pixel_for_lat_lng(matching_collection.cpcs[0], *boundary_br)
        cpc_width = exact_cpc_br[0] - exact_cpc_tl[0]
        cpc_height = exact_cpc_br[1] - exact_cpc_tl[1]

        cpc_scale = (cpc_width / xwidth, cpc_height / ywidth)
        cpc_offset = (exact_cpc_tl[0] - cpc_tl[0] + 0.5, exact_cpc_tl[1] - cpc_tl[1] + 0.5)

        countries = matching_collection.countries.read_array(xmin, ymin, xwidth, ywidth)
        points = np.zeros((ywidth, xwidth))
        for ypos in range(ywidth):
            for xpos in range(xwidth):
                if boundary[ypos, xpos] == 0:
                    continue
                ecoregion = ecoregions[ypos, xpos]
                country = countries[ypos, xpos]
                luc0 = lucs[0][ypos, xpos]
                luc5 = lucs[1][ypos, xpos]
                luc10 = lucs[2][ypos, xpos]
                key = build_key(ecoregion, country, luc0, luc5, luc10)
                if key in ktrees:
                    cpcx = math.floor((xpos + 4) * cpc_scale[0] + cpc_offset[0]) # Alignment WHY?
                    cpcy = math.floor((ypos - 6) * cpc_scale[1] + cpc_offset[1]) # WHY WHY WHY
                    points[ypos, xpos] = 1 if ktrees[key].contains(np.array([
                        elevations[ypos, xpos],
                        slopes[ypos, xpos],
                        accesses[ypos, xpos],
                        cpcs[0][cpcy, cpcx],
                        cpcs[1][cpcy, cpcx],
                        cpcs[2][cpcy, cpcx],
                        cpcs[3][cpcy, cpcx],
                        cpcs[4][cpcy, cpcx],
                        cpcs[5][cpcy, cpcx],
                    ])) else 0
        # Write points to output
        # pylint: disable-next=protected-access
        matching_pixels._dataset.GetRasterBand(1).WriteArray(points, xmin, ymin)
        print(f"Worker {worker_index} completed coords {coords}.")
    print(f"Worker {worker_index} finished.")

    # Ensure we flush pixels to disk now we're finished
    del matching_pixels._dataset


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
        coordinate_queue = manager.Queue()

        worker_count = processes_count

        # Fill the co-ordinate queue
        for ypos in range(DIVISIONS):
            coordinate_queue.put([ypos, 0])
        for _ in range(worker_count):
            coordinate_queue.put(None)

        ktree = load_k(k_filename, start_year)

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
            ktree,
            coordinate_queue,
        )) for index in range(worker_count)]
        for worker_process in workers:
            worker_process.start()

        while workers:
            candidates = [x for x in workers if not x.is_alive()]
            for candidate in candidates:
                candidate.join()
                if candidate.exitcode:
                    for victim in workers:
                        victim.kill()
                    sys.exit(candidate.exitcode)
                workers.remove(candidate)
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
