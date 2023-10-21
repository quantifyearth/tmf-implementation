import argparse
import glob
import math
import os
import re
import shutil
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from time import perf_counter_ns
from typing import List, Tuple
from numba import jit, void, int64, int32

import numpy as np
from osgeo import gdal  # type: ignore
from yirgacheffe.layers import Layer, RasterLayer, GroupLayer, TiledGroupLayer  # type: ignore
from yirgacheffe.window import Area, PixelScale  # type: ignore

from methods.common import LandUseClass
from methods.common.geometry import wgs_aspect_ratio_at

# Example filename: JRC_TMF_AnnualChange_v1_2011_AFR_ID37_N0_E40.tif
JRC_FILENAME_RE = re.compile(r".*_v1_(\d+)_.*_([NS]\d+)_([EW]\d+)\.tif")

GEOMETRY_SCALE_ADJUSTMENT = True

# Usual cache reduction
# Although we run over each tile twice and tiles are big, so give it a bit more than usual
gdal.SetCacheMax(1024 * 1024 * 1024)

def accumulate(cumrow, width, x_radius, result_width):
    return cumrow[x_radius - width:x_radius - width + result_width] - cumrow[x_radius + width:x_radius + width + result_width]    

def fine_circular_jrc_tile(jrc_tile: Layer, all_jrc: GroupLayer, result_filename: str, luc: int) -> None:
    # TODO: should we actually calculate this in case e.g. JRC changes resolution
    diameter = 66 # 2010m cicles when you include center pixel
    radius = diameter // 2
    radius2 = radius * radius

    layers = [all_jrc, jrc_tile]
    jrc_tile_area = RasterLayer.find_intersection(layers)
    for layer in layers:
        layer.set_window_for_intersection(jrc_tile_area)

    result_width = math.floor(jrc_tile.window.xsize)
    result_height = math.floor(jrc_tile.window.ysize)

    result_layer = RasterLayer.empty_raster_layer(
        jrc_tile_area,
        jrc_tile.pixel_scale, # type: ignore
        gdal.GDT_Float32,
        result_filename,
        jrc_tile.projection,
    )

    ys = np.arange(0, diameter + 1)

    rows = None
    last_x_radius = 0
    start_time = perf_counter_ns()
    for yoffset in range(result_height):
        if yoffset % 100 == 0:
            print(f"{os.path.basename(result_filename)}: {yoffset} of {result_height}")
            if yoffset == 100:
                print("Time per pixel (nanosecond):", (perf_counter_ns() - start_time) / yoffset / result_width)
                exit()
        if GEOMETRY_SCALE_ADJUSTMENT:
            x_factor = wgs_aspect_ratio_at(jrc_tile.latlng_for_pixel(0, yoffset)[0])
        else:
            x_factor = 1
        x_factor2 = x_factor * x_factor
        x_radius = math.ceil(radius * x_factor)
        x_diameter = 2 * x_radius
        circle_mask = np.zeros((diameter + 1, x_diameter + 1))
        change_at = np.array([0] * (diameter + 1))
        for y in range(-radius, radius + 1):
            on = False
            for x in range(-x_radius, x_radius + 1):
                r2 = y*y + x*x / x_factor2
                if r2 <= radius2:
                    circle_mask[y + radius, x + x_radius] = 1
                    if not on:
                        on = True
                        change_at[y + radius] = x
        row_list = np.unique(change_at)
        mask_sum = np.sum(circle_mask, dtype=np.float32)

        if rows is None or last_x_radius != x_radius:
            # Read in initial full width stripe of height DIAMETER + 1 (or new initial stripe if x_radius has changed)
            src = all_jrc.read_array(-x_radius, yoffset - radius, result_width + x_diameter + 1, diameter + 1)
            sums = np.cumsum(src == luc, axis = 1, dtype = np.int16)
            rows = [{width: accumulate(cumrow, width, x_radius, result_width) for width in row_list} for cumrow in sums]
        else:
            # Shift src up and add on next stripe of height 1
            next_row = all_jrc.read_array(-x_radius, yoffset + radius, result_width + x_diameter + 1, 1)
            cumrow = np.cumsum(next_row == luc, dtype = np.int16)
            next_rows = {width: accumulate(cumrow, width, x_radius, result_width) for width in row_list}
            rows = rows[1:] + [next_rows]
        last_x_radius = x_radius

        buffer = np.sum([rows[y][change_at[y]] for y in ys], axis=0, dtype=np.int16)

        result_layer._dataset.GetRasterBand(1).WriteArray(np.array([buffer / mask_sum]), 0, yoffset) # pylint: disable=W0212
    print(f"{os.path.basename(result_filename)} complete")

# As we move along the columns, for each row we remove the left-most point of the last circle,
# and add the right-most point of the new circle. Because circles don't have holes, this is sufficient
# to shift the circle over, and saves us many adds.
@jit(void(int64, int32[:, :], int64[:], int32, int32[:]), nopython=True, fastmath=True)
def do_running_sum(x_radius, src, change_at, running_sum, buffer):
    stride = src.shape[1]
    src = src.flatten()
    ys = np.arange(len(change_at))
    xoffs = x_radius + change_at - 1 + ys * stride
    xons = x_radius - change_at + ys * stride
    for xoffset in range(1, len(buffer)):
        for y in range(0, len(xoffs)):
            running_sum -= src[xoffset + xoffs[y]]
        for y in range(0, len(xons)):
            running_sum += src[xoffset + xons[y]]
        buffer[xoffset] = running_sum

def process_jrc_tiles_by_year_and_region(
    output_directory_path: str,
    temporary_directory: str,
    tilepaths: List[str]
) -> None:
    jrc_tiles = []
    for p in tilepaths:
        jrc_tiles.append(RasterLayer.layer_from_file(p))
    jrc_layer = GroupLayer(jrc_tiles)

    for p in tilepaths:
        for luc in [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED]:
            target_filename = f"fine_{os.path.splitext(os.path.basename(p))[0]}_{luc.value}.tif"
            target_path = os.path.join(output_directory_path, target_filename)
            if not os.path.exists(target_path):
                tempdest = os.path.join(temporary_directory, target_filename)
                fine_circular_jrc_tile(RasterLayer.layer_from_file(p), jrc_layer, tempdest, luc.value)
                shutil.move(tempdest, target_path)

def generate_fine_circular_coverage(
    jrc_directory: str,
    result_directory: str,
    concurrent_processes: int
) -> None:
    os.makedirs(result_directory, exist_ok=True)

    jrc_filenames = glob.glob("*.tif", root_dir=jrc_directory)
    jrc_file_paths = [os.path.join(jrc_directory, x) for x in jrc_filenames]
    years = list(set([x.split('_')[4] for x in jrc_filenames]))
    years.sort()

    # Split JRC into its three separate areas (separated by the Atlantic, Indian, and Pacific oceans)
    areas = ["SAM", "AFR", "ASI"]

    filesets = []
    for year in years:
        for area in areas:
            filesets.append([x for x in jrc_file_paths if f"/JRC_TMF_AnnualChange_v1_{year}" in x and f"_{area}_" in x])

    total_files = sum(len(files) for files in filesets)

    assert total_files == len(jrc_file_paths)

    with tempfile.TemporaryDirectory() as tempdir:
        with Pool(processes=concurrent_processes) as pool:
            pool.map(
                partial(
                    process_jrc_tiles_by_year_and_region,
                    result_directory,
                    tempdir,
                ),
                filesets
            )

def main() -> None:
    # If you use the default multiprocess model then you risk deadlocks when logging (which we
    # have hit). Spawn is the default on macOS, but not on Linux.
    set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Generate coarsened proportional coverage tiles from JRC Annual Change data."
    )
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC Annual Change GeoTIFF tiles."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_directory_path",
        help="Directory into which output GeoTIFF files will be written. Will be created if it does not exist."
    )
    parser.add_argument(
        "-j",
        type=int,
        required=False,
        default=round(cpu_count() / 2),
        dest="processes",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    generate_fine_circular_coverage(
        args.jrc_directory_path,
        args.output_directory_path,
        args.processes
    )

if __name__ == "__main__":
    main()
