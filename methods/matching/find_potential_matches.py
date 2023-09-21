import argparse
from collections import defaultdict
import glob
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue, cpu_count
import timeit

from osgeo import gdal  # type: ignore
import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection
from methods.common.luc import luc_matching_columns, luc_range

# We do not re-use data in this, so set a small block cache size for GDAL, otherwise
# it pointlessly hogs memory, and then spends a long time tidying it up after.
gdal.SetCacheMax(1024 * 1024 * 16)

@dataclass
class MatchedPixel:
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


def load_k(
    k_filename: str,
    sentinal_count: int,
    output_queue: Queue,
) -> None:
    # put the source pixels into the queue
    source_pixels = pd.read_parquet(k_filename)
    i = 0
    for row in source_pixels.iterrows():
        output_queue.put(row)
        i += 1
        if i == 10:
            break
    # To close the pipe put in one sentinel value per worker
    for _ in range(sentinal_count):
        output_queue.put(None)

def reduce_results(
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    start_year: int,
    evaluation_year: int,
    result_dataframe_filename: str,
    sentinal_count: int,
    input_queue: Queue
) -> None:
    # lazily created
    merged_result = None

    while True:
        partial_raster_filename = input_queue.get()
        if partial_raster_filename is None:
            sentinal_count -= 1
            if sentinal_count == 0:
                break
        else:
            partial_raster = RasterLayer.layer_from_file(partial_raster_filename)
            if merged_result is None:
                merged_result = partial_raster
            else:
                calc = merged_result + partial_raster
                temp = RasterLayer.empty_raster_layer_like(merged_result)
                calc.save(temp)
                merged_result = temp

    # merged result should be all the pixels now
    assert merged_result is not None
    matching_collection = build_layer_collection(
        merged_result.pixel_scale,
        merged_result.projection,
        list(luc_range(start_year, evaluation_year)),
        [start_year, start_year - 5, start_year - 10],
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
    )

    assert matching_collection.boundary.window == merged_result.window
    assert matching_collection.boundary.area == merged_result.area

    results = []

    # now we we need to scan for matched pixels and store the data about them
    width = matching_collection.boundary.window.xsize
    for yoffset in range(matching_collection.boundary.window.ysize):
        row_matches = merged_result.read_array(0, yoffset, width, 1)
        if row_matches.sum() == 0:
            continue
        row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
        row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
        row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
        row_access = matching_collection.access.read_array(0, yoffset, width, 1)
        row_lucs = [x.read_array(0, yoffset, width, 1) for x in matching_collection.lucs]
        # For CPC, which is at a different pixel_scale, we need to do a little math
        coord = matching_collection.boundary.latlng_for_pixel(0, yoffset)
        _, cpc_yoffset = matching_collection.cpcs[0].pixel_for_latlng(*coord)
        row_cpc = [
            cpc.read_array(0, cpc_yoffset, matching_collection.cpcs[0].window.xsize, 1)
            for cpc in matching_collection.cpcs
        ]

        for xoffset in range(width):
            if not row_matches[0][xoffset]:
                continue

            coord = matching_collection.boundary.latlng_for_pixel(xoffset, yoffset)
            cpc_xoffset, _ = matching_collection.cpcs[0].pixel_for_latlng(*coord)
            cpcs = [x[0][cpc_xoffset] for x in row_cpc]

            results.append([
                coord[0],
                coord[1],
                row_ecoregion[0][xoffset],
                row_elevation[0][xoffset],
                row_slope[0][xoffset],
                row_access[0][xoffset],
            ] + [luc[0][xoffset] for luc in row_lucs] + cpcs)

    luc_columns = [f'luc_{year}' for year in luc_range(start_year, evaluation_year)]
    cpc_columns = ['cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']
    output = pd.DataFrame(
        results,
        columns=['lat', 'lng', 'ecoregion', 'elevation', 'slope', 'access'] + luc_columns + cpc_columns
    )
    output.to_parquet(result_dataframe_filename)

def worker(
    _worker_index: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    start_year: int,
    _evaluation_year: int,
    result_folder: str,
    input_queue: Queue,
    output_queue: Queue,
) -> None:
    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    luc0, luc5, luc10 = luc_matching_columns(start_year)

    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        [start_year, start_year - 5, start_year - 10],
        [], # CPC not needed at this stage
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
    )

    while True:
        row = input_queue.get()
        if row is None:
            break
        index, matching = row

        result_path = os.path.join(result_folder, f"{index}.tif")
        if os.path.exists(result_path):
            try:
                raster = RasterLayer.layer_from_file(result_path)
                if raster.sum() > 0:
                    output_queue.put(result_path)
                    continue
            except FileNotFoundError:
                pass

        matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=result_path)

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
        filtered_luc0 = matching_collection.lucs[0].numpy_apply(lambda chunk: chunk == matching[luc0])
        filtered_luc5 = matching_collection.lucs[1].numpy_apply(lambda chunk: chunk == matching[luc5])
        filtered_luc10 = matching_collection.lucs[2].numpy_apply(lambda chunk: chunk == matching[luc10])

        calc = matching_collection.boundary * filtered_ecoregions * filtered_elevation * \
            filtered_luc0 * filtered_luc5 * filtered_luc10 * filtered_slopes * filtered_access
        count = calc.save(matching_pixels, and_sum=True)
        del matching_pixels._dataset
        if count > 0:
            output_queue.put(result_path)

    # Signal worker exited
    output_queue.put(None)

def find_potential_matches_old(
    k_filename: str,
    start_year: int,
    evaluation_year: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    result_folder: str,
    processes_count: int
) -> None:
    os.makedirs(result_folder, exist_ok=True)
    result_dataframe_filename = os.path.join(result_folder, "results.parquet")

    assert processes_count >= 3

    with Manager() as manager:
        source_queue = manager.Queue()
        results_queue = manager.Queue()

        worker_count = processes_count - 2
        consumer_process = Process(target=reduce_results, args=(
            matching_zone_filename,
            jrc_directory_path,
            cpc_directory_path,
            ecoregions_directory_path,
            elevation_directory_path,
            slope_directory_path,
            access_directory_path,
            start_year,
            evaluation_year,
            result_dataframe_filename,
            worker_count,
            results_queue
        ))
        consumer_process.start()
        workers = [Process(target=worker, args=(
            index,
            matching_zone_filename,
            jrc_directory_path,
            cpc_directory_path,
            ecoregions_directory_path,
            elevation_directory_path,
            slope_directory_path,
            access_directory_path,
            start_year,
            evaluation_year,
            result_folder,
            source_queue,
            results_queue
        )) for index in range(worker_count)]
        for worker_process in workers:
            worker_process.start()
        ingest_process = Process(target=load_k, args=(k_filename, worker_count, source_queue))
        ingest_process.start()

        processes = workers + [ingest_process, consumer_process]
        while processes:
            candidates = [x for x in processes if not x.is_alive()]
            for candidate in candidates:
                candidate.join()
                if candidate.exitcode:
                    for victim in processes:
                        victim.kill()
                    sys.exit(candidate.exitcode)
                processes.remove(candidate)
            time.sleep(1)

def finalise_results(
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    start_year: int,
    evaluation_year: int,
    result_path: str,
    result_dataframe_filename: str,
) -> None:
    merged_result = RasterLayer.layer_from_file(result_path)

    matching_collection = build_layer_collection(
        merged_result.pixel_scale,
        merged_result.projection,
        list(luc_range(start_year, evaluation_year)),
        [start_year, start_year - 5, start_year - 10],
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
    )

    assert matching_collection.boundary.window == merged_result.window
    assert matching_collection.boundary.area == merged_result.area

    results = []

    # now we we need to scan for matched pixels and store the data about them
    width = matching_collection.boundary.window.xsize
    for yoffset in range(matching_collection.boundary.window.ysize):
        row_matches = merged_result.read_array(0, yoffset, width, 1)
        if row_matches.sum() == 0:
            continue
        row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
        row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
        row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
        row_access = matching_collection.access.read_array(0, yoffset, width, 1)
        row_lucs = [x.read_array(0, yoffset, width, 1) for x in matching_collection.lucs]
        # For CPC, which is at a different pixel_scale, we need to do a little math
        coord = matching_collection.boundary.latlng_for_pixel(0, yoffset)
        _, cpc_yoffset = matching_collection.cpcs[0].pixel_for_latlng(*coord)
        row_cpc = [
            cpc.read_array(0, cpc_yoffset, matching_collection.cpcs[0].window.xsize, 1)
            for cpc in matching_collection.cpcs
        ]

        for xoffset in range(width):
            if not row_matches[0][xoffset]:
                continue

            coord = matching_collection.boundary.latlng_for_pixel(xoffset, yoffset)
            cpc_xoffset, _ = matching_collection.cpcs[0].pixel_for_latlng(*coord)
            cpcs = [x[0][cpc_xoffset] for x in row_cpc]

            results.append([
                coord[0],
                coord[1],
                row_ecoregion[0][xoffset],
                row_elevation[0][xoffset],
                row_slope[0][xoffset],
                row_access[0][xoffset],
            ] + [luc[0][xoffset] for luc in row_lucs] + cpcs)

    luc_columns = [f'luc_{year}' for year in luc_range(start_year, evaluation_year)]
    cpc_columns = ['cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']
    output = pd.DataFrame(
        results,
        columns=['lat', 'lng', 'ecoregion', 'elevation', 'slope', 'access'] + luc_columns + cpc_columns
    )
    output.to_parquet(result_dataframe_filename)

def find_potential_matches(
    k_filename: str,
    start_year: int,
    evaluation_year: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    result_folder: str,
    _processes_count: int
) -> None:
    luc0, luc5, luc10 = luc_matching_columns(start_year)
  
    source_pixels = pd.read_parquet(k_filename)
    # Split source_pixels into classes
    source_classes = defaultdict(list)
    elevation_range = [math.inf, -math.inf]
    slope_range = [math.inf, -math.inf]
    access_range = [math.inf, -math.inf]
    elevation_width = 200
    slope_width = 2.5
    access_width = 10
    for _, row in source_pixels.iterrows():
        key = (int(row.ecoregion) << 16) | (int(row[luc0]) << 10) | (int(row[luc5]) << 5) | (int(row[luc10]))
        source_classes[key].append(row)
        if row.elevation - elevation_width < elevation_range[0]: elevation_range[0] = row.elevation - elevation_width
        if row.elevation + elevation_width > elevation_range[1]: elevation_range[1] = row.elevation + elevation_width

        if row.slope - slope_width < slope_range[0]: slope_range[0] = row.slope - slope_width
        if row.slope + slope_width > slope_range[1]: slope_range[1] = row.slope + slope_width

        if row.access - access_width < access_range[0]: access_range[0] = row.access - access_width
        if row.access + access_width > access_range[1]: access_range[1] = row.access + access_width

    source_nps = dict()
    for key, values in source_classes.items():
        print(key, "\t", hex(key), len(values))
        source_nps[key] = np.array([(row.elevation, row.slope, row.access) for row in values])
    

    print(elevation_range)
    print(slope_range)
    print(access_range)

    os.makedirs(result_folder, exist_ok=True)
    result_dataframe_filename = os.path.join(result_folder, "results.parquet")
    result_path = os.path.join(result_folder, "results.tif")

    # everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
    example_jrc_filename = glob.glob("*.tif", root_dir=jrc_directory_path)[0]
    example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_directory_path, example_jrc_filename))

    matching_collection = build_layer_collection(
        example_jrc_layer.pixel_scale,
        example_jrc_layer.projection,
        [start_year, start_year - 5, start_year - 10],
        [], # CPC not needed at this stage
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
    )

    matching_pixels = RasterLayer.empty_raster_layer_like(matching_collection.boundary, filename=result_path)

    print(matching_collection.ecoregions.window)
    window = matching_collection.elevation.window
    print("Running")
    ysection = math.ceil(window.ysize / 10)
    xsection = math.ceil(window.xsize / 10)
    if (False):
        for yoff in range(0, window.ysize, ysection):
            ystep = ysection
            if yoff + ystep > window.ysize:
                ystep = window.ysize - yoff
            for xoff in range(0, window.xsize, xsection):
                xstep = xsection
                if xoff + xstep > window.xsize:
                    xstep = window.xsize - xoff
                print(xoff, yoff)
    else:
            xstep = xsection
            ystep = ysection
            yoff = ystep * 7
            xoff = xstep * 6
            ecoregions = matching_collection.ecoregions.read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            elevation = matching_collection.elevation.read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            slopes = matching_collection.slope.read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            access = matching_collection.access.read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            luc0_layer = matching_collection.lucs[0].read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            luc5_layer = matching_collection.lucs[1].read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            luc10_layer = matching_collection.lucs[2].read_array(window.xoff + xoff, window.yoff + yoff, xstep, ystep)
            print("Loaded")
            data = np.full(shape=(ystep, xstep), fill_value=0)
            for y in range(ystep):
                print("y:", y)
                for x in range(xstep):
                    key = (int(ecoregions[y, x]) << 16) | (int(luc0_layer[y, x]) << 10) | (int(luc5_layer[y, x]) << 5) | (int(luc10_layer[y, x]))
                    if key in source_nps:
                        sources = source_nps[key]
                        pos = np.where(elevation[y, x] + 200 < sources[:, 0], False,
                            np.where(elevation[y, x] - 200 > sources[:, 0], False,
                            np.where(slopes[y, x] + 2.5 < sources[:, 1], False,
                            np.where(slopes[y, x] - 2.5 > sources[:, 1], False,
                            np.where(access[y, x] + 10 < sources[:, 2], False,
                            np.where(access[y, x] - 10 > sources[:, 2], False,
                                     True
                        ))))))
                        data[y, x] = 1 if np.any(pos) else 0
            matching_pixels._dataset.WriteArray(data, xoff=xoff, yoff=yoff)
    print("WriteArray")
    exit()

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
    filtered_luc0 = matching_collection.lucs[0].numpy_apply(lambda chunk: chunk == matching[luc0])
    filtered_luc5 = matching_collection.lucs[1].numpy_apply(lambda chunk: chunk == matching[luc5])
    filtered_luc10 = matching_collection.lucs[2].numpy_apply(lambda chunk: chunk == matching[luc10])

    calc = matching_collection.boundary * filtered_ecoregions * filtered_elevation * \
        filtered_luc0 * filtered_luc5 * filtered_luc10 * filtered_slopes * filtered_access
    calc.save(matching_pixels)
    del matching_pixels._dataset

    finalise_results(
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
        start_year,
        evaluation_year,
        result_path,
        result_dataframe_filename,
    )


def main():
    parser = argparse.ArgumentParser(description="Finds all potential matches to K in matching zone, aka set S.")
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        dest="k_filename",
        help="Parquet file containing pixels from K as generated by calculate_k.py"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file desribing area from which matching pixels may be selected."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        dest="start_year",
        help="Year project started."
    )
    parser.add_argument(
        "--evaluation_year",
        type=int,
        required=True,
        dest="evaluation_year",
        help="Year of project evalation"
    )
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Directory containing JRC AnnualChange GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--cpc",
        type=str,
        required=True,
        dest="cpc_directory_path",
        help="Filder containing Coarsened Proportional Coverage GeoTIFF tiles for all years."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        dest="ecoregions_directory_path",
        help="Directory containing Ecoregions GeoTIFF tiles."
    )
    parser.add_argument(
        "--elevation",
        type=str,
        required=True,
        dest="elevation_directory_path",
        help="Directory containing SRTM elevation GeoTIFF tiles."
    )
    parser.add_argument(
        "--slope",
        type=str,
        required=True,
        dest="slope_directory_path",
        help="Directory containing slope GeoTIFF tiles."
    )
    parser.add_argument(
        "--access",
        type=str,
        required=True,
        dest="access_directory_path",
        help="Directory containing access to health care GeoTIFF tiles."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination parquet file for results."
    )
    parser.add_argument(
        "-j",
        type=int,
        required=False,
        default=round(cpu_count() / 2),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    find_potential_matches(
        args.k_filename,
        args.start_year,
        args.evaluation_year,
        args.matching_zone_filename,
        args.jrc_directory_path,
        args.cpc_directory_path,
        args.ecoregions_directory_path,
        args.elevation_directory_path,
        args.slope_directory_path,
        args.access_directory_path,
        args.output_filename,
        args.processes_count
    )

if __name__ == "__main__":
    main()
