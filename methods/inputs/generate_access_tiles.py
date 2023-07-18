import glob
import os
import re
import shutil
import sys
import tempfile
from functools import partial
from multiprocessing import Pool

from yirgacheffe.layers import RasterLayer  # type: ignore

def process_file(
    access_geotif_filename: str,
    output_folder: str,
    jrc_filename: str,
) -> None:
    matches = re.match(r".*_([NS]\d+)_([WE]\d+).tif", jrc_filename)
    assert matches is not None
    result_filename = f"access_{matches[1]}_{matches[2]}.tif"
    result_path = os.path.join(output_folder, result_filename)
    if os.path.exists(result_path):
        return

    access_data = RasterLayer.layer_from_file(access_geotif_filename)

    with tempfile.TemporaryDirectory() as tempdir:

        jrc_raster = RasterLayer.layer_from_file(jrc_filename)
        access_data.set_window_for_intersection(jrc_raster.area)

        temp = RasterLayer.empty_raster_layer_like(access_data)
        access_data.save(temp)

        temp_output = os.path.join(tempdir, result_filename)

        scaled = RasterLayer.scaled_raster_from_raster(temp, jrc_raster.pixel_scale, filename=temp_output)
        del scaled._dataset

        shutil.move(temp_output, result_path)

def generate_access_tiles(
    access_geotif_filename: str,
    jrc_folder: str,
    output_folder: str
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    jrc_files = [os.path.join(jrc_folder, filename) for filename in glob.glob("*2020*.tif", root_dir=jrc_folder)]

    with Pool(processes=50) as pool:
        pool.map(
            partial(
                process_file,
                access_geotif_filename,
                output_folder,
            ),
            jrc_files
        )

def main() -> None:
    try:
        access_geotif_filename = sys.argv[1]
        jrc_folder = sys.argv[2]
        output_folder = sys.argv[3]
    except IndexError:
        print(f"Usage: {sys.argv[0]} ACCESS_GEOTIFF JRC_FOLDER OUTPUT_FOLDER", file=sys.stderr)
        sys.exit(1)

    generate_access_tiles(access_geotif_filename, jrc_folder, output_folder)

if __name__ == "__main__":
    main()
