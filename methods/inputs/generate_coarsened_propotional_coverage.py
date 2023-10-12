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

import numpy as np
from osgeo import gdal  # type: ignore
from yirgacheffe.layers import RasterLayer, GroupLayer, TiledGroupLayer  # type: ignore
from yirgacheffe.window import Area, PixelScale  # type: ignore

from methods.common import LandUseClass

# Example filename: JRC_TMF_AnnualChange_v1_2011_AFR_ID37_N0_E40.tif
JRC_FILENAME_RE = re.compile(r".*_v1_(\d+)_.*_([NS]\d+)_([EW]\d+)\.tif")

def coarsened_jrc_tile(jrc_layer: TiledGroupLayer, result_filename: str, luc: int) -> None:

    result_pixel_scale = PixelScale(jrc_layer.pixel_scale.xstep * 40, jrc_layer.pixel_scale.ystep * 40)
    result_width = math.floor(jrc_layer.window.xsize / 40.0)
    result_height = math.floor(jrc_layer.window.ysize / 40.0)
    result_area = Area(
        top=jrc_layer.area.top,
        left=jrc_layer.area.left,
        bottom=jrc_layer.area.top + (result_height * result_pixel_scale.ystep),
        right=jrc_layer.area.left + (result_width * result_pixel_scale.xstep),
    )

    result_layer = RasterLayer.empty_raster_layer(
        result_area,
        result_pixel_scale,
        gdal.GDT_Float32,
        result_filename,
        jrc_layer.projection
    )

    # Temp hack to move to from area to point pixel alignment until we add support to yirgacheffe
    # for pixel vs area offsets.
    result_layer._dataset.SetGeoTransform(
        (
            result_layer.area.left + (result_pixel_scale.xstep / 2),
            result_pixel_scale.xstep,
            0.0,
            result_layer.area.top + (result_pixel_scale.ystep / 2),
            0.0,
            result_pixel_scale.ystep,
        )
    )

    for yoffset in range(result_height):
        buffer = []
        src = jrc_layer.read_array(0, yoffset * 40, result_width * 40, 40)
        filtered = np.asarray(src==luc)
        for xoffset in range(result_width):
            subset = filtered[0:40, xoffset*40:(xoffset+1)*40]
            buffer.append(subset.sum() / (40.0 * 40.0))
        result_layer._dataset.GetRasterBand(1).WriteArray(np.array([buffer]), 0, yoffset) # pylint: disable=W0212

def process_jrc_tiles_by_year(
    output_directory_path: str,
    temporary_directory: str,
    fileinfo: Tuple[str,List[str]]
) -> None:
    year, tilepaths = fileinfo
    jrc_layer = GroupLayer([RasterLayer.layer_from_file(x) for x in tilepaths])
    for luc in [LandUseClass.UNDISTURBED, LandUseClass.DEFORESTED]:
        target_filename = f"coarse_{year}_{luc.value}.tif"
        target_path = os.path.join(output_directory_path, target_filename)
        if not os.path.exists(target_path):
            tempdest = os.path.join(temporary_directory, target_filename)
            coarsened_jrc_tile(jrc_layer, tempdest, luc.value)
            shutil.move(tempdest, target_path)

def generate_coarsened_proportional_coverage(
    jrc_directory: str,
    result_directory: str,
    concurrent_processes: int
) -> None:
    os.makedirs(result_directory, exist_ok=True)

    jrc_filenames = glob.glob("*.tif", root_dir=jrc_directory)
    years = list(set([x.split('_')[4] for x in jrc_filenames]))
    years.sort()

    filesets = []
    for year in years:
        year_files = [x for x in jrc_filenames if year in x]
        filesets.append((year, [os.path.join(jrc_directory, x) for x in year_files]))


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
        # process_jrc_tiles_by_year(result_directory, tempdir, filesets[-1])

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
