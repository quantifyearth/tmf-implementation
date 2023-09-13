import argparse
import glob
import os

from osgeo.gdal import GDT_Int32
from yirgacheffe.layers import RasterLayer, VectorLayer

def generate_country_raster(
    jrc_directory_path: str,
    matching_zone_filename: str,
    countries_shape_filename: str,
    output_filename: str
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    matching = VectorLayer.layer_from_file(
        matching_zone_filename,
        None,
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
    )

    countries = VectorLayer.layer_from_file(
        countries_shape_filename,
        None,
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        datatype=GDT_Int32,
        burn_value="osm_id"
    )

    countries.set_window_for_intersection(matching.area)
    result = RasterLayer.empty_raster_layer_like(countries, output_filename, compress=False)
    countries.save(result)


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
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file desribing area from which matching pixels may be selected."
    )
    parser.add_argument(
        "--countries",
        type=str,
        required=True,
        dest="countries_shape_filename",
        help="GeoJSON of country boundaries."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination raster file."
    )
    args = parser.parse_args()

    generate_country_raster(
        args.jrc_directory_path,
        args.matching_zone_filename,
        args.countries_shape_filename,
        args.output_filename,
    )

if __name__ == "__main__":
    main()
