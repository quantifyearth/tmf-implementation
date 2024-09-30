import argparse
import glob
import os
import math
from typing import Dict, List

import geopandas as gpd  # type: ignore
import numpy as np
import pandas as pd
from scipy import stats 
from yirgacheffe.layers import RasterLayer, GroupLayer  # type: ignore

def generate_carbon_density(
    jrc_directory_path: str,
    gedi_data_file: str,
    output_file: str
) -> None:

    # validate the output file extension
    _, output_ext = os.path.splitext(output_file)
    output_ext = output_ext.lower()
    if output_ext not in ['.csv', '.parquet']:
        raise ValueError("We only support .csv and .parquet outputs.")

    # load the jrc raster layers for 2020
    jrc_raster_layer = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename)) for filename in
            glob.glob("*2020*.tif", root_dir=jrc_directory_path)
    ], name="luc_2020")

    # read the gedi data into a geodataframe
    gedi = gpd.read_file(gedi_data_file)

    # get the pixel scale from the raster layer
    pixel_scale = jrc_raster_layer.pixel_scale

    # initialize a dictionary to hold agbd values for each land use class
    luc_buckets: Dict[int, List[float]] = {}

    # iterate over each gedi point
    for _, row in gedi.iterrows():
        agbd = row['agbd']
        point = row['geometry']

        # compute the x and y offsets in the raster grid
        xoffset = math.floor((point.x - jrc_raster_layer.area.left) / pixel_scale.xstep)
        yoffset = math.floor((point.y - jrc_raster_layer.area.top) / pixel_scale.ystep)

        # check the JRC data around this cell
        surroundings = jrc_raster_layer.read_array(xoffset - 1, yoffset - 1, 3, 3)

        # get the land use class at the center
        land_use_class = surroundings[1][1]

        # check if all surrounding cells have the same land use class
        if not np.all(surroundings == land_use_class):
            continue

        # add the agbd value to the corresponding land use class bucket
        luc_buckets.setdefault(land_use_class, []).append(agbd)

    # get a sorted list of land use classes
    land_use_class_list = sorted(luc_buckets.keys())

    # prepare a list to collect results
    results = []
    for land_use_class in land_use_class_list:
        # get the list of agbd values for this land use class
        agbd_values = luc_buckets[land_use_class]
        # compute the median agbd
        median_agbd = np.median(agbd_values)
        # append the results to the list
        results.append([
            land_use_class,
            median_agbd
        ])

    # create a dataframe from the results
    output = pd.DataFrame(results, columns=[
        "luc",
        "carbon_density
    ])

    # write the output to a file
    match output_ext:
        case '.csv':
            output.to_csv(output_file, index=False)
        case '.parquet':
            output.to_parquet(output_file)
        case _:
            assert False, "Extensions was validated earlier"

def main() -> None:
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculates carbon density for different land use classes")
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="Location of JRC tiles."
    )
    parser.add_argument(
        "--gedi",
        type=str,
        required=True,
        dest="gedi_data_file",
        help="Location of filtered GEDI data."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Output file name."
    )
    args = parser.parse_args()

    # generate carbon density data
    generate_carbon_density(
        args.jrc_directory_path,
        args.gedi_data_file,
        args.output_filename)

if __name__ == "__main__":
    main()