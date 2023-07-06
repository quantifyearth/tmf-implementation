import glob
import os
import shutil
import sys
import tempfile
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection

def find_potential_matches_per_k_item(
    matching_zone_filename: str,
    jrc_data_folder: str,
    ecoregions_folder_filename: str,
    elevation_folder_filename: str,
    slope_folder_filename: str,
    access_folder_filename: str,
    year: int,
    result_folder: str,
    matching_row: Tuple
) -> Tuple[int,int]:
    index, matching = matching_row

    output_filename = f"{index}.tif"
    output_path = os.path.join(result_folder, output_filename)

    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_data_folder)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_data_folder, example_jrc_filename))

    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        year,
        matching_zone_filename,
        jrc_data_folder,
        ecoregions_folder_filename,
        elevation_folder_filename,
        slope_folder_filename,
        access_folder_filename,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        temp_output = os.path.join(tempdir, output_filename)
        matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=temp_output)

        filtered_ecoregions = matching_collection.ecoregions.numpy_apply(lambda chunk: chunk == matching.ecoregion)
        filtered_elevation = matching_collection.elevation.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.elevation - 200), chunk <= (matching.elevation + 200))
        )
        filtered_slopes = matching_collection.slope.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.slope - 2.5), chunk <= (matching.slope + 2.5))
        )
        filtered_access = matching_collection.access.numpy_apply(
            lambda chunk: np.logical_and(chunk >= (matching.access - 10), chunk <= (matching.access + 10))
        )
        filtered_luc0 = matching_collection.lucs[0].numpy_apply(lambda chunk: chunk == matching.luc0)
        filtered_luc5 = matching_collection.lucs[1].numpy_apply(lambda chunk: chunk == matching.luc5)
        filtered_luc10 = matching_collection.lucs[2].numpy_apply(lambda chunk: chunk == matching.luc10)

        calc = matching_collection.boundary * filtered_ecoregions * filtered_elevation * \
            filtered_luc0 * filtered_luc5 * filtered_luc10 * filtered_slopes * filtered_access
        count = calc.save(matching_pixels, and_sum=True)
        del matching_pixels._dataset
        if count == 0:
            print(f"No matching pixels found for index {index}")
        else:
            shutil.move(temp_output, output_path)

    # now we have in matching_pixels just those that match, so we now need to store the data about them
    return (index, count)

def find_potential_matches(
    k_filename: str,
    matching_zone_filename: str,
    jrc_data_folder: str,
    ecoregions_folder_filename: str,
    elevation_folder_filename: str,
    slope_folder_filename: str,
    access_folder_filename: str,
    year: int,
    result_folder: str,
) -> None:
    os.makedirs(result_folder, exist_ok=True)

    source_pixels = pd.read_parquet(k_filename)
    k_pixels = list(source_pixels.iterrows())

    with Pool(processes=200) as pool:
        res = pool.map(
            partial(
                find_potential_matches_per_k_item,
                matching_zone_filename,
                jrc_data_folder,
                ecoregions_folder_filename,
                elevation_folder_filename,
                slope_folder_filename,
                access_folder_filename,
                year,
                result_folder
            ),
            k_pixels
        )
        print(f"zero results: {len([x for x in res if x[1] == 0])}")

def main():
    try:
        k_filename = sys.argv[1]
        matching_zone_filename = sys.argv[2]
        jrc_data_folder = sys.argv[3]
        ecoregions_folder_filename = sys.argv[4]
        elevation_folder_filename = sys.argv[5]
        slope_folder_filename = sys.argv[6]
        access_folder_filename = sys.argv[7]
        year = int(sys.argv[8])
        result_folder = sys.argv[9]
    except (IndexError, ValueError):
        print(f"Usage: {sys.argv[0]} K_PARQUET MATCHING_ZONE JRC_FOLDER "
            "ECOREGIONS_FOLDER ELEVATION_FOLDER SLOPES_FOLDER ACCESS_FOLDER YEAR OUT",
            file=sys.stderr)
        sys.exit(1)

    find_potential_matches(
        k_filename,
        matching_zone_filename,
        jrc_data_folder,
        ecoregions_folder_filename,
        elevation_folder_filename,
        slope_folder_filename,
        access_folder_filename,
        year,
        result_folder
    )

if __name__ == "__main__":
    main()