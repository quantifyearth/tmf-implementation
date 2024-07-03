# This script is used to slim down the ecoregions dataset, which is otherwise quite heavy to work with.
# Running this step is not strictly necessary, but is useful in interactive development to make things
# a bit responsive, but is not needed for the production pipeline.

import glob
import os
import re
import sys

import shapely # type: ignore
from geopandas import gpd # type: ignore
from shapely.geometry import Polygon # type: ignore

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

# JRC_TMF_AnnualChange_v1_2021_ASI_ID42_N0_E130.tif
JRC_RE = re.compile(r".*_([NS])(\d+)_([WE])(\d+)\.tif")

def generate_jrc_mask(jrc_files_directory: str) -> gpd.GeoSeries:
    tile_set = set()
    for filename in glob.glob("*.tif", root_dir=jrc_files_directory):
        match = JRC_RE.match(filename)
        if match is None:
            raise ValueError(f"Failed to parse JRC filename {filename}")
        dir_lat, val_lat, dir_lng, val_lng = match.groups()
        top = int(val_lat) * (1.0 if dir_lat == 'N' else -1.0)
        left = int(val_lng) * (1.0 if dir_lng == 'E' else -1.0)
        tile_set.add((top, left))

    tiles = []
    for top, left in tile_set:
        bottom = top - 10.0
        right = left + 10.0

        tiles.append(Polygon([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
            [left, top]
        ]))

    all_tiles = gpd.GeoSeries(tiles)
    merged_files = shapely.unary_union(all_tiles)
    return gpd.GeoSeries(merged_files)

def simplify_ecoregions(source_path: str, output_filename: str, jrc_files_directory: str) -> None:
    tiles = generate_jrc_mask(jrc_files_directory)

    raw_ecoregions = gpd.read_file(source_path)

    # Find just the regions that overlap areas where there are JRC tiles
    jrc_limited = raw_ecoregions[raw_ecoregions.intersects(tiles[0])]

    # This clause removes the ice/rock ecoregion which includes the large areas of land, including
    # Antarctic and Greenland. It gets included as some of the JRC tiles just touch Tibet. Removing
    # this has a significant benefit in terms of responsiveness of GeoJSON processing.
    no_ice = jrc_limited[jrc_limited['ECO_ID'] != 0.0]

    # This file is still big enough that the user will need to set OGR_GEOJSON_MAX_OBJ_SIZE, but
    # it's less than half the raw file size.
    no_ice.to_file(output_filename, driver="GeoJSON")

def main() -> None:
    try:
        source_path = sys.argv[1]
        jrc_files_directory = sys.argv[3]
        target_path = sys.argv[2]
    except IndexError:
        print(f"Usage: {sys.argv[0]} SOURCE_PATH TARGET_PATH JRC_TILES", file=sys.stderr)
        sys.exit(1)

    simplify_ecoregions(source_path, target_path, jrc_files_directory)

if __name__ == "__main__":
    main()
