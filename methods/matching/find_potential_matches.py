import glob
import os
import sys

import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection

def find_potential_matches(
    k_filename: str,
    k_row: int,
    matching_zone_filename: str,
    jrc_data_folder: str,
    ecoregions_folder_filename: str,
    elevation_folder_filename: str,
    year: int,
    result_filename
) -> None:
    source_pixels = pd.read_parquet(k_filename)
    matching = source_pixels.iloc[k_row]

    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_data_folder)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_data_folder, example_jrc_filename))

    print("Loading layers")
    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        year,
        matching_zone_filename,
        jrc_data_folder,
        ecoregions_folder_filename,
        elevation_folder_filename,
    )

    matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=result_filename)

    filtered_ecoregions = matching_collection.ecoregions.numpy_apply(lambda chunk: chunk == matching.ecoregion)
    filtered_elevation = matching_collection.elevation.numpy_apply(
        lambda chunk: np.logical_and(chunk >= (matching.elevation - 200), chunk <= (matching.elevation - 200))
    )
    filtered_luc0 = matching_collection.lucs[0].numpy_apply(lambda chunk: chunk == matching.luc0)
    filtered_luc5 = matching_collection.lucs[1].numpy_apply(lambda chunk: chunk == matching.luc5)
    filtered_luc10 = matching_collection.lucs[2].numpy_apply(lambda chunk: chunk == matching.luc10)

    print("calculating")
    calc = matching_collection.boundary * filtered_ecoregions * filtered_elevation * \
        filtered_luc0 * filtered_luc5 * filtered_luc10
    count = calc.save(matching_pixels, and_sum=True)
    print(f"Found {count} matches")

def main():
    try:
        k_filename = sys.argv[1]
        k_row = int(sys.argv[2])
        matching_zone_filename = sys.argv[3]
        jrc_data_folder = sys.argv[4]
        ecoregions_folder_filename = sys.argv[5]
        elevation_folder_filename = sys.argv[6]
        year = int(sys.argv[7])
        result_filename = sys.argv[8]
    except (IndexError, ValueError):
        print(f"Usage: {sys.argv[0]} K_PARQUET ITEM MATCHING_ZONE JRC_FOLDER ECOREGIONS_FOLDER ELEVATION_FOLDER YEAR OUT",
            file=sys.stderr)
        sys.exit(1)

    find_potential_matches(
        k_filename,
        k_row,
        matching_zone_filename,
        jrc_data_folder,
        ecoregions_folder_filename,
        elevation_folder_filename,
        year,
        result_filename
    )

if __name__ == "__main__":
    main()