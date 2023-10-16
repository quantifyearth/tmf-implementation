import argparse
import glob
import math
import os
import re
import shutil
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from typing import List, Tuple
from numba import jit, void, int64, float32, boolean

import numpy as np
from osgeo import gdal  # type: ignore
from yirgacheffe.layers import Layer, RasterLayer, GroupLayer, TiledGroupLayer  # type: ignore
from yirgacheffe.window import Area, PixelScale  # type: ignore

from methods.common import LandUseClass
from methods.common.geometry import wgs_aspect_ratio_at

# Example filename: JRC_TMF_AnnualChange_v1_2011_AFR_ID37_N0_E40.tif
JRC_FILENAME_RE = re.compile(r".*_v1_(\d+)_.*_([NS]\d+)_([EW]\d+)\.tif")

GEOMETRY_SCALE_ADJUSTMENT = False

def fine_circular_jrc_tile(jrc_tile: Layer, all_jrc: GroupLayer, result_filename: str, luc: int) -> None:
    diameter = 32 # 960m cicles; TODO: should we actually calculate this in case e.g. JRC changes resolution
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

    src = None
    last_x_radius = 0
    for yoffset in range(result_height):
        if yoffset % 100 == 0:
            print(f"{os.path.basename(result_filename)}: {yoffset} of {result_height}")
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

        mask_sum = np.sum(circle_mask, dtype=np.float32)

        if src is None or last_x_radius != x_radius:
            # Read in initial full width stripe of height DIAMETER + 1 (or new initial stripe if x_radius has changed)
            src = all_jrc.read_array(-x_radius, yoffset - radius, result_width + x_diameter + 1, diameter + 1)
            src = np.asarray(src == luc, dtype=np.float32)
        else:
            # Shift src up and add on next stripe of height 1
            next_row = all_jrc.read_array(-x_radius, yoffset + radius, result_width + x_diameter + 1, 1)
            src_without_first_row = src[1:]
            src = np.concatenate([src_without_first_row, next_row == luc], axis=0, dtype=np.float32)
        last_x_radius = x_radius

        # Form the initial circle
        initial = src[0:diameter + 1, 0:x_diameter + 1]
        initial_circle = initial * circle_mask

        running_sum = np.sum(initial_circle, dtype=np.float32)

        buffer = np.zeros(result_width, dtype=np.float32)
        buffer[0] = running_sum / mask_sum

        do_running_sum(radius, x_radius, result_width, src, change_at, mask_sum, running_sum, buffer)

        result_layer._dataset.GetRasterBand(1).WriteArray(np.array([buffer]), 0, yoffset) # pylint: disable=W0212
    print(f"{os.path.basename(result_filename)} complete")

# As we move along the row, for each column we remove the left-most point of the last circle,
# and add the right-most point of the new circle. Because circles don't have holes, this is sufficient
# to shift the circle over, and saves us many adds.
@jit(void(int64, int64, int64, float32[:, :], int64[:], float32, float32, float32[:]), nopython=True, fastmath=True)
def do_running_sum(radius, x_radius, result_width, src, change_at, mask_sum, running_sum, buffer):
    for xoffset in range(1, result_width):
        for y in range(0, radius * 2 + 1):
            running_sum -= src[y, xoffset + x_radius + change_at[y] - 1]
            running_sum += src[y, xoffset + x_radius - change_at[y]]
        buffer[xoffset] = running_sum / mask_sum


def process_jrc_tiles_by_year(
    output_directory_path: str,
    temporary_directory: str,
    fileinfo: Tuple[str,List[str]]
) -> None:
    year, tilepaths = fileinfo
    print(year)
    jrc_tiles = []
    for p in tilepaths:
        jrc_tiles.append(RasterLayer.layer_from_file(p))
    jrc_layer = GroupLayer(jrc_tiles)

    for luc in [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED]:
        for p in tilepaths:
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

    filesets = []
    for year in years:
        filesets.append((year, [x for x in jrc_file_paths if year in x]))

    with tempfile.TemporaryDirectory() as tempdir:
        with Pool(processes=concurrent_processes) as pool:
            pool.map(
                partial(
                    process_jrc_tiles_by_year,
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
