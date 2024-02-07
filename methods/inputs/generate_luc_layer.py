import argparse
import math
import os
import re
import sys
from typing import List

from osgeo import gdal, ogr # type: ignore
from yirgacheffe.layers import RasterLayer, VectorLayer  # type: ignore

REPRESENTATIVE_YEAR = 2020

def directed_to_regular_degrees(notated: str) -> int:
    """Converts the NSEW prepended degrees in the filename to signed ints"""
    direction = notated[0]
    value = int(notated[1:])
    if direction in ['S', 'W']:
        value *= -1
    return value

def get_jrc_paths(layer: ogr.Layer, annual_change_path: str) -> List[str]:
    # work out region for mask
    envelopes = []
    layer.ResetReading()
    feature = layer.GetNextFeature()
    while feature:
        envelopes.append(feature.GetGeometryRef().GetEnvelope())
        feature = layer.GetNextFeature()
    if len(envelopes) == 0:
        raise ValueError('No geometry found')

    first_x = math.floor(min(x[0] for x in envelopes) / 10.0) * 10
    last_x = math.floor(max(x[1] for x in envelopes) / 10.0) * 10
    first_y = math.ceil(max(x[3] for x in envelopes) / 10.0) * 10
    last_y = math.ceil(min(x[2] for x in envelopes) / 10.0) * 10

    # the JRC data is sparse, so not all files might exist, so we take what we can
    results = []
    filename_re = re.compile(r"JRC_TMF_AnnualChange_v1_" + str(REPRESENTATIVE_YEAR) + \
        r"_\w+_ID\d+_([NS]\d+)_([EW]\d+).tif")
    for path in os.listdir(annual_change_path):
        match = filename_re.match(path)
        if not match:
            continue
        directed_y, directed_x = match.groups()

        tile_x = directed_to_regular_degrees(directed_x)
        tile_y = directed_to_regular_degrees(directed_y)

        if (first_x <= tile_x <= last_x) and (first_y >= tile_y >= last_y):
            results.append(os.path.join(annual_change_path, path))

    return results

def generate_luc_layer(boundary_filename: str, jrc_folder: str, luc_raster_filename: str) -> None:
    boundary_dataset = ogr.Open(boundary_filename)
    if boundary_dataset is None:
        print(f"Failed to open {boundary_filename}", file=sys.stderr)
        sys.exit(1)
    boundary_layer = boundary_dataset.GetLayer()

    # Get the JRC layers that make up this area
    jrc_paths = get_jrc_paths(boundary_layer, jrc_folder)
    if len(jrc_paths) == 0:
        print("No JRC tiles found for area", file=sys.stderr)
        sys.exit(1)

    jrc_layer_set = [RasterLayer.layer_from_file(x) for x in jrc_paths]

    if len(jrc_layer_set) > 1:
        # To work around a limitation in yirgacheffe, which I'll fix shortly but I want to get this
        # code in first as this is more important, we have to render the JRC tiles to a large canvas, and then shrink it
        # down again
        jrc_union_area = RasterLayer.find_union(jrc_layer_set)
        for layer in jrc_layer_set:
            layer.set_window_for_union(jrc_union_area)

        super_jrc_raster = RasterLayer.empty_raster_layer_like(jrc_layer_set[0])

        def _numpy_merge(lhs, rhs):
            lhs[rhs>0] = rhs[rhs>0]
            return lhs

        calc = jrc_layer_set[0]
        for layer in jrc_layer_set[1:]:
            calc = calc.numpy_apply(_numpy_merge, layer)
        calc.save(super_jrc_raster)
    else:
        super_jrc_raster = jrc_layer_set[0]

    # load the boundary as a vector layer just to get its area
    boundary_vector = VectorLayer(boundary_layer, jrc_layer_set[0].pixel_scale, jrc_layer_set[0].projection)
    # pad the area by a pixel on each side, as we need to get surrounding pixels for any pixel within the boundary
    boundary_vector.offset_window_by_pixels(1)
    target_raster = RasterLayer.empty_raster_layer_like(boundary_vector, filename=luc_raster_filename)

    super_jrc_raster.set_window_for_intersection(target_raster.area)
    super_jrc_raster.save(target_raster)

def main() -> None:
    # We do not re-use data in this, so set a small block cache size for GDAL, otherwise
    # it pointlessly hogs memory, and then spends a long time tidying it up after.
    gdal.SetCacheMax(1024 * 1024 * 16)

    parser = argparse.ArgumentParser(description="Generate LUC tile for project.")
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        dest="boundary_filename",
        help="Buffer boundary."
    )
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Location of JRC tiles."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Project specific LUC tif."
    )
    args = parser.parse_args()

    generate_luc_layer(args.boundary_filename, args.jrc_directory_path, args.output_filename)

if __name__ == "__main__":
    main()
