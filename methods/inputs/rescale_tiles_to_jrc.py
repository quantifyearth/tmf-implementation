import argparse
import glob
import os
import shutil
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count

from yirgacheffe.layers import RasterLayer  # type: ignore
from yirgacheffe.window import PixelScale  # type: ignore

def process_tile(
    tile_directory_path: str,
    output_directory_path: str,
    tempdir: str,
    pixel_scale: PixelScale,
    tile: str,
) -> None:
    tile_path = os.path.join(tile_directory_path, tile)
    rescaled_name = os.path.join(tempdir, tile)
    target_name = os.path.join(output_directory_path, tile)

    rescaled = RasterLayer.scaled_raster_from_raster(
        RasterLayer.layer_from_file(tile_path),
        pixel_scale,
        filename=rescaled_name,
        compress=False
    )
    # evil force GDAL to flush
    del rescaled._dataset
    shutil.move(rescaled_name, target_name)

def rescale_tiles_to_jrc(
    jrc_directory_path: str,
    tile_directory_path: str,
    output_directory_path: str,
    processes_count: int
) -> None:
    os.makedirs(output_directory_path, exist_ok=True)

    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    with tempfile.TemporaryDirectory() as tempdir:
        tiles = glob.glob("*.tif", root_dir=tile_directory_path)
        with Pool(processes=processes_count) as pool:
            pool.map(
                partial(
                    process_tile,
                    tile_directory_path,
                    output_directory_path,
                    tempdir,
                    example_jrc_layer.pixel_scale,
                ),
                tiles
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculates sample pixels in project, aka set K")
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--tiles",
        type=str,
        required=True,
        dest="tile_directory_path",
        help="Directory containing tiles to rescale"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_directory_path",
        help="Destination directory for new tiles"
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

    rescale_tiles_to_jrc(
        args.jrc_directory_path,
        args.tile_directory_path,
        args.output_directory_path,
        args.processes_count,
    )

if __name__ == "__main__":
    main()
