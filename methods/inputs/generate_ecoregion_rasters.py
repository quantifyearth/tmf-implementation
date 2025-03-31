import argparse
import glob
import os
import shutil
import tempfile

from osgeo import gdal # type: ignore
from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer  # type: ignore

# The real cost here is repeated re-drawing of the ecoregions, so cut down on that
import yirgacheffe.operators  # type: ignore
yirgacheffe.operators.YSTEP = 1024 * 8

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def generate_ecoregion_rasters(
    jrc_directory_path: str,
    matching_zone_filename: str,
    ecoregions_filename: str,
    output_filename: str
) -> None:
    output_folder, _ = os.path.split(output_filename)
    os.makedirs(output_folder, exist_ok=True)

    jrc_layer = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename))
        for filename in glob.glob("*2020*.tif", root_dir=jrc_directory_path)
    ])

    matching = VectorLayer.layer_from_file(
        matching_zone_filename,
        None,
        jrc_layer.pixel_scale,
        jrc_layer.projection,
    )

    ecoregions = VectorLayer.layer_from_file(
        ecoregions_filename,
        None,
        jrc_layer.pixel_scale,
        jrc_layer.projection,
        datatype=gdal.GDT_UInt16,
        burn_value="ECO_ID"
    )

    layers = [jrc_layer, matching, ecoregions]
    intersection = RasterLayer.find_intersection(layers)
    for layer in layers:
        layer.set_window_for_intersection(intersection)

    with tempfile.TemporaryDirectory() as tempdir:
        target_filename = os.path.join(tempdir, "ecoregion.tif")

        result = RasterLayer.empty_raster_layer_like(jrc_layer, filename=target_filename,
            datatype=ecoregions.datatype, compress=False)
        ecoregions.save(result)
        result.close()
        shutil.move(target_filename, output_filename)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GeoTIFF from ecoregion data")
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file describing area from which matching pixels may be selected."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        dest="ecoregions_filename",
        help="GeoJSON of ecoregions."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination raster file folder."
    )
    args = parser.parse_args()

    generate_ecoregion_rasters(
        args.jrc_directory_path,
        args.matching_zone_filename,
        args.ecoregions_filename,
        args.output_filename,
    )

if __name__ == "__main__":
    main()
