import glob
import os
import sys
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue
from typing import Tuple

import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore
import yirgacheffe.operators

from methods.matching.calculate_k import build_layer_collection

yirgacheffe.operators.YSTEP = 1024 * 12

@dataclass
class MatchedPixel:
    k_lat: float
    k_lng: float
    match_lat: float
    match_lng: float
    xoffset: int
    yoffset: int
    ecoregion: int
    elevation: float
    slope: float
    access: float
    luc0: float

    def __eq__(self, val):
        return (self.xoffset, self.yoffset) == val

    def __ne__(self, val):
        return (self.xoffset, self.yoffset) != val

    def __hash__(self):
        return (self.xoffset, self.yoffset).__hash__()


def threads() -> int:
    return 100


def load_k(k_filename: str, queue: Queue) -> None:
    # put the source pixels into the queue
    source_pixels = pd.read_parquet(k_filename)
    for _, row in source_pixels.iterrows():
        queue.put(row)
    # To close the pipe put in one sentinel value per worker
    for _ in range(threads()):
        queue.put(None)

def reduce_results(
    result_dataframe_filename: str,
    queue: Queue
) -> None:
    sentinal_count = threads()
    results = set()
    while True:
        vals = queue.get()
        if vals is None:
            sentinal_count -= 1
            if sentinal_count == 0:
                break
        else:
            results.update(vals)

    output = pd.DataFrame(results)
    output.to_parquet(result_dataframe_filename)

def worker(
    worker_index: int,
    matching_zone_filename: str,
    jrc_data_folder: str,
    ecoregions_folder_filename: str,
    elevation_folder_filename: str,
    slope_folder_filename: str,
    access_folder_filename: str,
    start_year: int,
    _evaluation_year: int,
    input_queue: Queue,
    output_queue: Queue,
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_data_folder)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_data_folder, example_jrc_filename))

    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        start_year,
        matching_zone_filename,
        jrc_data_folder,
        ecoregions_folder_filename,
        elevation_folder_filename,
        slope_folder_filename,
        access_folder_filename,
    )

    while True:
        matching = input_queue.get()
        if matching is None:
            break

        matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary)

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

        if count > 0:
            # now we we need to scan for matched pixels and store the data about them

            width = matching_collection.boundary.window.xsize
            row_results = []
            for yoffset in range(matching_collection.boundary.window.ysize):
                row_matches = matching_pixels.read_array(0, yoffset, width, 1)
                if row_matches.sum() == 0:
                    continue
                row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
                row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
                row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
                row_access = matching_collection.access.read_array(0, yoffset, width, 1)
                row_luc0 = matching_collection.lucs[0].read_array(0, yoffset, width, 1)
                for xoffset in range(width):
                    if not row_matches[0][xoffset]:
                        continue
                    row_results.append(MatchedPixel(
                        matching.lat,
                        matching.lng,
                        matching_pixels.area.top + (yoffset * matching_pixels.pixel_scale.ystep),
                        matching_pixels.area.left + (xoffset * matching_pixels.pixel_scale.xstep),
                        xoffset,
                        yoffset,
                        row_ecoregion[0][xoffset],
                        row_elevation[0][xoffset],
                        row_slope[0][xoffset],
                        row_access[0][xoffset],
                        row_luc0[0][xoffset],
                    ))
                if len(row_results) > 1000:
                    output_queue.put(row_results)
                    row_results = []

    # Signal worker exited
    output_queue.put(None)

def find_potential_matches(
    k_filename: str,
    matching_zone_filename: str,
    jrc_data_folder: str,
    ecoregions_folder_filename: str,
    elevation_folder_filename: str,
    slope_folder_filename: str,
    access_folder_filename: str,
    start_year: int,
    evaluation_year: int,
    result_folder: str,
) -> None:
    os.makedirs(result_folder, exist_ok=True)
    result_dataframe_filename = os.path.join(result_folder, "results.parquet")

    with Manager() as manager:
        source_queue = manager.Queue()
        results_queue = manager.Queue()

        consumer_process = Process(target=reduce_results, args=(
            result_dataframe_filename,
            results_queue,
        ))
        consumer_process.start()
        workers = [Process(target=worker, args=(
            index,
            matching_zone_filename,
            jrc_data_folder,
            ecoregions_folder_filename,
            elevation_folder_filename,
            slope_folder_filename,
            access_folder_filename,
            start_year,
            evaluation_year,
            source_queue,
            results_queue
        )) for index in range(threads())]
        for worker_process in workers:
            worker_process.start()
        ingest_process = Process(target=load_k, args=(k_filename, source_queue))
        ingest_process.start()

        ingest_process.join()
        for worker_process in workers:
            worker_process.join()
        consumer_process.join()

def main():
    try:
        k_filename = sys.argv[1]
        matching_zone_filename = sys.argv[2]
        jrc_data_folder = sys.argv[3]
        ecoregions_folder_filename = sys.argv[4]
        elevation_folder_filename = sys.argv[5]
        slope_folder_filename = sys.argv[6]
        access_folder_filename = sys.argv[7]
        start_year = int(sys.argv[8])
        evaluation_year = int(sys.argv[9])
        result_folder = sys.argv[10]
    except (IndexError, ValueError):
        print(f"Usage: {sys.argv[0]} K_PARQUET MATCHING_ZONE JRC_FOLDER "
            "ECOREGIONS_FOLDER ELEVATION_FOLDER SLOPES_FOLDER ACCESS_FOLDER "
            "START_YEAR EVALUATION_YEAR OUT",
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
        start_year,
        evaluation_year,
        result_folder
    )

if __name__ == "__main__":
    main()
