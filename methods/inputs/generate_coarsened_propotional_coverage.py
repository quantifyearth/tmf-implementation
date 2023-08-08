import argparse
import glob
import math
import os
import re
import shutil
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
from osgeo import gdal  # type: ignore
from yirgacheffe.layers import RasterLayer  # type: ignore
from yirgacheffe.window import Area, PixelScale  # type: ignore

from methods.common import LandUseClass

# Example filename: JRC_TMF_AnnualChange_v1_2011_AFR_ID37_N0_E40.tif
JRC_FILENAME_RE = re.compile(r".*_v1_(\d+)_.*_([NS]\d+)_([EW]\d+)\.tif")

def coarsened_jrc_tile(tile_filename: str, result_filename: str, luc: int) -> None:
    # We rely on the fact that the JRC files are actually slightly overlapped and
    # just ignore the fact the boundaries won't quite line up for now. In theory though we should
    # be using multiple tiles to generate this
    src = RasterLayer.layer_from_file(tile_filename)
    filtered_result = RasterLayer.empty_raster_layer_like(src)
    def _filter(chunk):
        return np.asarray(chunk==luc)
    calc = src.numpy_apply(_filter)
    calc.save(filtered_result)

    result_pixel_scale = PixelScale(src.pixel_scale.xstep * 40, src.pixel_scale.ystep * 40)
    result_width = math.floor(src.window.xsize / 40.0)
    result_height = math.floor(src.window.ysize / 40.0)
    result_area = Area(
        top=src.area.top,
        left=src.area.left,
        bottom=src.area.top + (result_height * result_pixel_scale.ystep),
        right=src.area.left + (result_height * result_pixel_scale.xstep),
    )

    result_layer = RasterLayer.empty_raster_layer(
        result_area,
        result_pixel_scale,
        gdal.GDT_Float32,
        result_filename,
        src.projection
    )

    for yoffset in range(result_height):
        buffer = []
        src = filtered_result.read_array(0, yoffset * 40, result_width * 40, 40)
        for xoffset in range(result_width):
            subset = src[0:40, xoffset*40:(xoffset+1)*40]
            buffer.append(subset.sum() / (40.0 * 40.0))
        result_layer._dataset.GetRasterBand(1).WriteArray(np.array([buffer]), 0, yoffset) # pylint: disable=W0212

def process_jrc_tile(
    output_directory_path: str,
    temporary_directory: str,
    jrc_tile_filename: str
) -> None:
    match = JRC_FILENAME_RE.match(jrc_tile_filename)
    if match is None:
        raise ValueError(f"Failed to parse JRC filename {jrc_tile_filename}")
    year, xoffset, yoffset = match.groups()
    for luc in [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED]:
        target_filename = f"coarse_{xoffset}_{yoffset}_{year}_{luc.value}.tif"
        target_path = os.path.join(output_directory_path, target_filename)
        if not os.path.exists(target_path):
            tempdest = os.path.join(temporary_directory, target_filename)
            coarsened_jrc_tile(jrc_tile_filename, tempdest, luc.value)
            shutil.move(tempdest, target_path)

def generate_coarsened_proportional_coverage(
    jrc_directory: str,
    result_directory: str,
    concurrent_processes: int
) -> None:
    os.makedirs(result_directory, exist_ok=True)

    with tempfile.TemporaryDirectory() as tempdir:
        jrc_filenames = glob.glob("*.tif", root_dir=jrc_directory)
        with Pool(processes=concurrent_processes) as pool:
            pool.map(
                partial(
                    process_jrc_tile,
                    result_directory,
                    tempdir,
                ),
                [os.path.join(jrc_directory, x) for x in jrc_filenames]
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

    generate_coarsened_proportional_coverage(
        args.jrc_directory_path,
        args.output_directory_path,
        args.processes
    )

if __name__ == "__main__":
    main()
