import argparse
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


# main function to filter GEDI data based on the buffer and granules .csv
def filter_gedi_data(
    gedi_directory_path: str,
    buffer: str,
    csv: str,
    output: str
) -> None:

    # load the boundary file (e.g., shapefile or geojson)
    boundary = gpd.read_file(buffer)

    # specify the columns that are important for AGB calculations
    interesting_columns = {'sensitivity', 'geometry', 'absolute_time', 'l4_quality_flag', 'degrade_flag',
                           'beam_type', 'agbd'}

    beams_list = []  # list to collect data from all granules

    # load granule names from the provided .csv file
    granule_df = pd.read_csv(csv, sep=',', quotechar='"', usecols=['granule_name'])
    granule_files = granule_df['granule_name'].tolist()

    # loop through each granule name from the file
    for granule_name in granule_files:
        granule_path = os.path.join(gedi_directory_path, granule_name)  # create the full path to the granule
        if os.path.exists(granule_path):  # check if the granule file exists
            df = parse_file(granule_path)  # parse the granule file into a DataFrame of beams

            # drop any columns not in the interesting_columns set
            df.drop(list(set(df.columns) - interesting_columns), axis=1)

            # filter beams based on beam type, quality flag, degradation flag, and date range
            df.drop(df.index[df.beam_type != "full"], inplace=True)
            df.drop(df.index[df.degrade_flag != 0], inplace=True)
            df.drop(df.index[df.l4_quality_flag != 1], inplace=True)
            df.drop(df.index[df.absolute_time < datetime(2020, 1, 1, tzinfo=timezone.utc)], inplace=True)
            df.drop(df.index[df.absolute_time >= datetime(2021, 1, 1, tzinfo=timezone.utc)], inplace=True)

            # filter beams on whether the data point falls within the buffer geometry
            df.drop(df.index[~df.within(boundary.geometry.values[0])], inplace=True)

            beams_list.append(df)  # add the filtered beams DataFrame to the list

        # add error for missing GEDI
        else:
            raise ValueError("No granule found - check your GEDI folder for: ", granule_path)

    # concatenate all filtered granules into a single GeoDataFrame
    superset = gpd.GeoDataFrame(pd.concat(beams_list, ignore_index=True))
    # write the result to a GeoJSON file
    superset.to_file(output, driver="GeoJSON")

# main function to handle argument parsing and script execution
def main() -> None:
    # set up argument parsing
    parser = argparse.ArgumentParser(description="Takes a set of GEDI granules and filters down"
                                                 "The data to just what is needed for the AGB calculation")
    # add argument for the directory containing GEDI granules
    parser.add_argument(
        "--granules",
        type=str,
        required=True,
        dest="gedi_directory_path",
        help="Location of gedi granules"
    )
    # add argument for the buffer boundary file
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        help="Buffer boundary file"
    )
    # add argument for the .csv file containing the list of granules to process
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help=".csv file containing list of granules"
    )
    # add argument for the output file name
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file name"
    )
    args = parser.parse_args()  # parse the arguments from the command line

    # call the filter function with the provided arguments
    filter_gedi_data(
        args.gedi_directory_path,
        args.buffer,
        args.csv,
        args.output
    )

# entry point for the script
if __name__ == "__main__":
    main()  # run the main function if the script is executed