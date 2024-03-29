import glob
import os
import re
import shutil
import sys
import tempfile
from functools import partial
from multiprocessing import Pool

from osgeo import gdal # type: ignore
from yirgacheffe.layers import RasterLayer, VectorLayer  # type: ignore

# The real cost here is repeated re-drawing of the ecoregions, so cut down on that
import yirgacheffe.operators  # type: ignore
yirgacheffe.operators.YSTEP = 1024 * 8

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def process_tile(result_path: str, ecoregions_filename: str, jrc_path: str, ) -> None:
    matches = re.match(r".*_([NS]\d+)_([WE]\d+).tif", jrc_path)
    assert matches is not None
    filename = f"ecoregion_{matches[1]}_{matches[2]}.tif"
    final_filename = os.path.join(result_path, filename)

    try:
        raster = RasterLayer.layer_from_file(final_filename)
        if raster.sum() > 0:
            return
    except (FileNotFoundError, TypeError):
        # File doesn't exist or is corrupt, so we need to do the work
        pass

    with tempfile.TemporaryDirectory() as tempdir:
        jrc_raster = RasterLayer.layer_from_file(jrc_path)
        ecoregions = VectorLayer.layer_from_file(
            ecoregions_filename,
            None,
            jrc_raster.pixel_scale,
            jrc_raster.projection,
            datatype=gdal.GDT_UInt16,
            burn_value="ECO_ID"
        )

        target_filename = os.path.join(tempdir, filename)
        result = RasterLayer.empty_raster_layer_like(jrc_raster, filename=target_filename,
            datatype=ecoregions.datatype, compress=False)
        ecoregions.set_window_for_intersection(jrc_raster.area)
        ecoregions.save(result)
        del result._dataset
        shutil.move(target_filename, final_filename)

def generate_ecoregion_rasters(
    ecoregions_filename: str,
    jrc_folder: str,
    output_folder: str,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    jrc_files = [os.path.join(jrc_folder, filename) for filename in glob.glob("*2020*.tif", root_dir=jrc_folder)]
    with Pool(processes=50) as pool:
        pool.map(partial(process_tile, output_folder, ecoregions_filename), jrc_files)

def main() -> None:
    try:
        ecoregions_filename = sys.argv[1]
        jrc_folder = sys.argv[2]
        output_folder = sys.argv[3]
    except IndexError:
        print(f"Usage: {sys.argv[0]} ECOREGIONS_GEOJSON JRC_FOLDER OUTPUT_FOLDER", file=sys.stderr)
        sys.exit(1)

    generate_ecoregion_rasters(ecoregions_filename, jrc_folder, output_folder)

if __name__ == "__main__":
    main()
