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
        raise ValueError("we only support .csv and .parquet outputs.")

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

        # read a 3x3 array around the point
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
        # compute the number of points
        num_points = len(agbd_values)
        # compute the median agbd
        median_agbd = np.median(agbd_values)
        # compute the 25th percentile agbd
        lower_quartile_agbd = np.percentile(agbd_values, 25)
        # compute the 75th percentile agbd
        upper_quartile_agbd = np.percentile(agbd_values, 75)
        # compute the mean agbd
        mean_agbd = np.mean(agbd_values)
        # compute the standard deviation of agbd
        std_dev_agbd = np.std(agbd_values, ddof=1)  # sample standard deviation
        # compute the standard error
        standard_error = std_dev_agbd / np.sqrt(num_points)
        # degrees of freedom
        df = num_points - 1
        # t-critical value for 95% confidence interval
        t_critical = stats.t.ppf(0.975, df)  # for 95% confidence
        # margin of error
        margin_of_error = t_critical * standard_error
        # compute confidence interval for mean agbd
        ci_lower_agbd = mean_agbd - margin_of_error
        ci_upper_agbd = mean_agbd + margin_of_error

        # convert median agbd to carbon density
        bgbd_median = median_agbd * 0.2  # below-ground biomass density
        deadwood_bd_median = median_agbd * 0.11  # deadwood biomass density
        total_median = median_agbd + bgbd_median + deadwood_bd_median
        carbon_density_median = total_median * 0.47  # convert biomass to carbon density

        # convert lower quartile agbd to carbon density
        bgbd_lower = lower_quartile_agbd * 0.2
        deadwood_bd_lower = lower_quartile_agbd * 0.11
        total_lower = lower_quartile_agbd + bgbd_lower + deadwood_bd_lower
        carbon_density_lower = total_lower * 0.47

        # convert upper quartile agbd to carbon density
        bgbd_upper = upper_quartile_agbd * 0.2
        deadwood_bd_upper = upper_quartile_agbd * 0.11
        total_upper = upper_quartile_agbd + bgbd_upper + deadwood_bd_upper
        carbon_density_upper = total_upper * 0.47

        # convert mean agbd to carbon density
        bgbd_mean = mean_agbd * 0.2
        deadwood_bd_mean = mean_agbd * 0.11
        total_mean = mean_agbd + bgbd_mean + deadwood_bd_mean
        carbon_density_mean = total_mean * 0.47

        # convert confidence interval bounds to carbon density
        bgbd_ci_lower = ci_lower_agbd * 0.2
        deadwood_bd_ci_lower = ci_lower_agbd * 0.11
        total_ci_lower = ci_lower_agbd + bgbd_ci_lower + deadwood_bd_ci_lower
        carbon_density_ci_lower = total_ci_lower * 0.47

        bgbd_ci_upper = ci_upper_agbd * 0.2
        deadwood_bd_ci_upper = ci_upper_agbd * 0.11
        total_ci_upper = ci_upper_agbd + bgbd_ci_upper + deadwood_bd_ci_upper
        carbon_density_ci_upper = total_ci_upper * 0.47

        # compute standard deviation of carbon density
        # since the conversion factors are linear, we can apply the same factors
        # total biomass multiplier: 1 + 0.2 + 0.11 = 1.31
        # carbon density multiplier: 1.31 * 0.47 = 0.6157
        std_dev_carbon_density = std_dev_agbd * 0.6157

        # append the results to the list
        results.append([
            land_use_class,
            carbon_density_median,
            carbon_density_lower,
            carbon_density_upper,
            carbon_density_mean,
            carbon_density_ci_lower,
            carbon_density_ci_upper,
            std_dev_carbon_density,
            num_points
        ])

    # create a dataframe from the results
    output = pd.DataFrame(results, columns=[
        "luc",
        "carbon_density_median",
        "carbon_density_lower",
        "carbon_density_upper",
        "carbon_density_mean",
        "carbon_density_ci_lower",
        "carbon_density_ci_upper",
        "std_dev_carbon_density",
        "num_points"
    ])

    # write the output to a file
    match output_ext:
        case '.csv':
            output.to_csv(output_file, index=False)
        case '.parquet':
            output.to_parquet(output_file)
        case _:
            assert False, "extensions was validated earlier"

def main() -> None:
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="calculates carbon density for different land use classes")
    parser.add_argument(
        "--jrc",
        type=str,
        required=True,
        dest="jrc_directory_path",
        help="location of jrc tiles."
    )
    parser.add_argument(
        "--gedi",
        type=str,
        required=True,
        dest="gedi_data_file",
        help="location of filtered gedi data."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="output file name."
    )
    args = parser.parse_args()

    # generate carbon density data
    generate_carbon_density(
        args.jrc_directory_path,
        args.gedi_data_file,
        args.output_filename)

if __name__ == "__main__":
    main()
