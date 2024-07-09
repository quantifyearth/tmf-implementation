import argparse
import glob
import os
import warnings
from datetime import datetime, timezone

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
from biomassrecovery.data.gedi_granule import GediGranule  # type: ignore
from biomassrecovery.constants import WGS84  # type: ignore

def parse_file(file: str) -> gpd.GeoDataFrame:
    with GediGranule(file) as granule:
        granule_data = []
        for beam in granule.iter_beams():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    beam.sql_format_arrays()
                    granule_data.append(beam.main_data)
            except KeyError:
                pass
        df = pd.concat(granule_data, ignore_index=True)
        gdf = gpd.GeoDataFrame(df, crs=WGS84)
    return gdf

def filter_gedi_data(
    boundary_file: str,
    gedi_directory_path: str,
    output_file: str
) -> None:

    boundary = gpd.read_file(boundary_file)

    # These are the columns we care about for calculating AGB values, the rest we'll drop
    # to save memory/disk space
    interesting_columns = {'sensitivity', 'geometry', 'absolute_time', 'l4_quality_flag', 'degraded_flag',
        'beam_type', 'agbd'}

    granules = []
    for path in glob.glob("*.h5", root_dir=gedi_directory_path):
        df = parse_file(os.path.join(gedi_directory_path, path))

        # Drop data we don't care about
        # The date limits here come from discussion with Miranda Lam:
        # "we currently use the JRC Annual Change 2020 for land cover class and GEDI shots from 2020/1/1 to 2021/1/1"
        df.drop(list(set(df.columns) - interesting_columns), axis=1)
        df.drop(df.index[df.beam_type != "full"], inplace=True)
        df.drop(df.index[df.degrade_flag != 0], inplace=True)
        df.drop(df.index[df.l4_quality_flag != 1], inplace=True)
        df.drop(df.index[df.absolute_time < datetime(2020, 1, 1, tzinfo=timezone.utc)], inplace=True)
        df.drop(df.index[df.absolute_time >= datetime(2021, 1, 1, tzinfo=timezone.utc)], inplace=True)
        df.drop(df.index[df.within(boundary.geometry.values[0]) is False], inplace=True)

        granules.append(df)

    superset = gpd.GeoDataFrame(pd.concat(granules, ignore_index=True))
    superset.to_parquet(output_file)

def main() -> None:
    parser = argparse.ArgumentParser(description="Takes a set of GEDI granules and filters down" \
        " the data to just what is needed for the AGB calculation")
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        dest="buffer_boundary_filename",
        help="Buffer boundary file"
    )
    parser.add_argument(
        "--gedi",
        type=str,
        required=True,
        dest="gedi_directory_path",
        help="Location of GEDI Granules."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Output file name."
    )
    args = parser.parse_args()

    filter_gedi_data(
        args.buffer_boundary_filename,
        args.gedi_directory_path,
        args.output_filename
    )

if __name__ == "__main__":
    main()
