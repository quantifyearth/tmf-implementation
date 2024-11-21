import argparse
import glob
import os
import shutil
import tempfile

from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer  # type: ignore

def generate_access_tiles(
    jrc_directory_path: str,
    matching_zone_filename: str,
    access_geotif_filename: str,
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

    access_data = RasterLayer.layer_from_file(access_geotif_filename)
    access_data.set_window_for_intersection(matching.area)

    temp = RasterLayer.empty_raster_layer_like(access_data)
    access_data.save(temp)

    with tempfile.TemporaryDirectory() as tempdir:
        temp_output = os.path.join(tempdir, "rescaled.tif")

        scaled = RasterLayer.scaled_raster_from_raster(temp, jrc_layer.pixel_scale, filename=temp_output)
        scaled.close()

        shutil.move(temp_output, output_filename)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GeoTIFF from country boundaries")
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
        "--access",
        type=str,
        required=True,
        dest="access_geotif_filename",
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

    generate_access_tiles(
        args.jrc_directory_path,
        args.matching_zone_filename,
        args.access_geotif_filename,
        args.output_filename
    )

if __name__ == "__main__":
    main()
