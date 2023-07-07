import glob
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

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
    matching_row: Tuple
) -> Tuple[int,int,List]:
    index, matching = matching_row

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

    matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary)#, filename=temp_output)

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
    results = []
    if count == 0:
        print(f"No matching pixels found for index {index}")
    else:
        # now we we need to scan for matched pixels and store the data about them

        width = matching_collection.boundary.window.xsize
        for yoffset in range(matching_collection.boundary.window.ysize):
            row_matches = matching_pixels.read_array(0, yoffset, width, 1)
            if row_matches.sum() == 0:
                continue
            row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
            row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
            row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
            row_access = matching_collection.access.read_array(0, yoffset, width, 1)
            row_luc = [
                luc.read_array(0, yoffset, width, 1) for luc in matching_collection.lucs
            ]
            for xoffset in range(width):
                if not row_matches[0][xoffset]:
                    continue
                lucs = [x[0][xoffset] for x in row_luc]
                results.append([
                    xoffset,
                    yoffset,
                    matching_pixels.area.top + (yoffset * matching_pixels.pixel_scale.ystep),
                    matching_pixels.area.left + (xoffset * matching_pixels.pixel_scale.xstep),
                    row_elevation[0][xoffset],
                    row_slope[0][xoffset],
                    row_ecoregion[0][xoffset],
                    row_access[0][xoffset],
                ] + lucs)

    # now we have in matching_pixels just those that match, so we now need to store the data about them
    return (index, count, results)

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
    if len(k_pixels) == 0:
        raise ValueError("No pixels in K")

    with Pool(processes=200) as pool:
        results = pool.map(
            partial(
                find_potential_matches_per_k_item,
                matching_zone_filename,
                jrc_data_folder,
                ecoregions_folder_filename,
                elevation_folder_filename,
                slope_folder_filename,
                access_folder_filename,
                year,
            ),
            k_pixels
        )
        print(f"zero results: {len([x for x in results if x[1] == 0])}")

        # now we need to join all of the individual matches into one dataframe
        columns=['x', 'y', 'lat', 'lng', 'elevation', 'slope', 'ecoregion', 'access', 'luc0', 'luc5', 'luc10']

        # I suspect we're going to struggle with memory here, so just incrementally build the final array so we
        # don't end up with two copies of everything
        matches = pd.DataFrame(results[0][2], columns=columns)
        for res in results[1:]:
            next_frame = pd.DataFrame(res[2], columns=columns)
            matches = pd.concat([matches, next_frame]).drop_duplicates(['x', 'y'])
        print(f"Total potential match pixels: {len(matches)}")
        result_dataframe_filename = os.path.join(result_folder, "results.parquet")
        matches.to_parquet(result_dataframe_filename)


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