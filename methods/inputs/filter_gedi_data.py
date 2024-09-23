import argparse
import glob
import os
import warnings
from datetime import datetime, timezone

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
from biomassrecovery.data.gedi_granule import GediGranule  # type: ignore
from biomassrecovery.constants import WGS84  # type: ignore

# function to parse a single GEDI granule file and return a GeoDataFrame
def parse_file(file: str) -> gpd.GeoDataFrame:
    with GediGranule(file) as granule:
        granule_data = []
        for beam in granule.iter_beams():  # iterate over beams within the granule
            try:
                # suppress warnings while processing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    beam.sql_format_arrays()  # format arrays for the beam data
                    granule_data.append(beam.main_data)  # collect main data for the beam
            except KeyError:
                # if a KeyError occurs, skip this beam
                pass
        # concatenate beam data into a DataFrame
        df = pd.concat(granule_data, ignore_index=True)
        # convert to GeoDataFrame using WGS84 coordinates
        gdf = gpd.GeoDataFrame(df, crs=WGS84)
    return gdf

# function to load granule filenames from a text file
def load_granules_from_file(granule_list_file: str) -> list:
    """loads the granule filenames from a text file"""
    with open(granule_list_file, "r") as file:
        # read each line as a granule name and return the list
        granule_files = file.read().splitlines()
    return granule_files

# main function to filter GEDI data based on the buffer and granule list
def filter_gedi_data(
    gedi_directory_path: str,
    boundary_file: str,
    granule_list_file: str,
    output_filename: str
) -> None:

    # load the boundary file (e.g., shapefile or geojson)
    boundary = gpd.read_file(boundary_file)

    # specify the columns that are important for AGB calculations
    interesting_columns = {'sensitivity', 'geometry', 'absolute_time', 'l4_quality_flag', 'degrade_flag',
                           'beam_type', 'agbd'}

    granules = []  # list to collect data from all granules
    # load granule names from the provided text file
    granule_files = load_granules_from_file(granule_list_file)

    # loop through each granule name from the file
    for granule_name in granule_files:
        granule_path = os.path.join(gedi_directory_path, granule_name)  # create the full path to the granule
        if os.path.exists(granule_path):  # check if the granule file exists
            df = parse_file(granule_path)  # parse the granule file into a DataFrame

            # drop any columns not in the interesting_columns set
            df.drop(list(set(df.columns) - interesting_columns), axis=1)

            # filter rows based on beam type, quality flag, degradation flag, and date range
            df.drop(df.index[df.beam_type != "full"], inplace=True)
            df.drop(df.index[df.degrade_flag != 0], inplace=True)
            df.drop(df.index[df.l4_quality_flag != 1], inplace=True)

            # filter based on whether the data point falls within the boundary geometry
            df.drop(df.index[~df.within(boundary.geometry.values[0])], inplace=True)

            granules.append(df)  # add the filtered DataFrame to the list

    # concatenate all filtered granules into a single GeoDataFrame
    superset = gpd.GeoDataFrame(pd.concat(granules, ignore_index=True))
    # write the result to a GeoJSON file
    superset.to_file(output_filename, driver="GeoJSON")

# main function to handle argument parsing and script execution
def main() -> None:
    # set up argument parsing
    parser = argparse.ArgumentParser(description="takes a set of gedi granules and filters down"
                                                 " the data to just what is needed for the agb calculation")
    # add argument for the directory containing GEDI granules
    parser.add_argument(
        "--granules",
        type=str,
        required=True,
        dest="gedi_directory_path",
        help="location of gedi granules"
    )
    # add argument for the buffer boundary file
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        dest="buffer_boundary_filename",
        help="buffer boundary file"
    )
    # add argument for the text file containing the list of granules to process
    parser.add_argument(
        "--granule-list",
        type=str,
        required=True,
        dest="granule_list_file",
        help="text file containing list of granules"
    )
    # add argument for the output file name
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="output file name"
    )
    args = parser.parse_args()  # parse the arguments from the command line

    # call the filter function with the provided arguments
    filter_gedi_data(
        args.gedi_directory_path,
        args.buffer_boundary_filename,
        args.granule_list_file,
        args.output_filename
    )

# entry point for the script
if __name__ == "__main__":
    main()  # run the main function if the script is executed

