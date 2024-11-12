import argparse
import logging
from multiprocessing import cpu_count

import polars as pl
from yirgacheffe.layers import RasterLayer  # type: ignore

from methods.matching.calculate_k import build_layer_collection
from methods.common.luc import luc_range

def build_m_table(
    m_raster_path: str,
    lagged: bool,
    start_year: int,
    evaluation_year: int,
    matching_zone_filename: str,
    jrc_directory_path: str,
    cpc_directory_path: str,
    ecoregions_directory_path: str,
    elevation_directory_path: str,
    slope_directory_path: str,
    access_directory_path: str,
    countries_raster_filename: str,
    result_dataframe_filename: str,
    _processes_count: int
) -> None:

    merged_raster = RasterLayer.layer_from_file(m_raster_path)

    if lagged:
        luc_range_list = list(luc_range(start_year - 10, start_year))
        lag_year = 10
    else:
        luc_range_list = list(luc_range(start_year, evaluation_year))
        lag_year = 0

    matching_collection = build_layer_collection(
        merged_raster.pixel_scale,
        merged_raster.projection,
        luc_range_list,
        [start_year - lag_year, start_year - 5 - lag_year, start_year - 10 - lag_year],
        matching_zone_filename,
        jrc_directory_path,
        cpc_directory_path,
        ecoregions_directory_path,
        elevation_directory_path,
        slope_directory_path,
        access_directory_path,
        countries_raster_filename,
    )

    assert matching_collection.boundary.window == merged_raster.window
    assert matching_collection.boundary.area == merged_raster.area

    results = []
    if lagged:
        luc_columns = [f'luc_{year}' for year in luc_range(start_year - 10, start_year)]
    else:
        luc_columns = [f'luc_{year}' for year in luc_range(start_year, evaluation_year)]
    cpc_columns = ['cpc0_u', 'cpc0_d', 'cpc5_u', 'cpc5_d', 'cpc10_u', 'cpc10_d']
    columns = ['lat', 'lng', 'ecoregion', 'elevation', 'slope', 'access', 'country'] + luc_columns + cpc_columns

    # now we we need to scan for matched pixels and store the data about them
    width = matching_collection.boundary.window.xsize
    for yoffset in range(matching_collection.boundary.window.ysize):
        print(f"{yoffset}/{matching_collection.boundary.window.ysize}")
        row_matches = merged_raster.read_array(0, yoffset, width, 1)
        if row_matches.sum() == 0:
            continue
        row_ecoregion = matching_collection.ecoregions.read_array(0, yoffset, width, 1)
        row_countries = matching_collection.countries.read_array(0, yoffset, width, 1)
        row_elevation = matching_collection.elevation.read_array(0, yoffset, width, 1)
        row_slope = matching_collection.slope.read_array(0, yoffset, width, 1)
        row_access = matching_collection.access.read_array(0, yoffset, width, 1)
        row_lucs = [x.read_array(0, yoffset, width, 1) for x in matching_collection.lucs]
        row_cpcs = [x.read_array(0, yoffset, width, 1) for x in matching_collection.cpcs]

        for xoffset in range(width):
            if not row_matches[0][xoffset]:
                continue

            coord = matching_collection.boundary.latlng_for_pixel(xoffset, yoffset)
            results.append([
                coord[0],
                coord[1],
                row_ecoregion[0][xoffset],
                row_elevation[0][xoffset],
                row_slope[0][xoffset],
                row_access[0][xoffset],
                row_countries[0][xoffset],
           ] + [luc[0][xoffset] for luc in row_lucs] + [cpc[0][xoffset] for cpc in row_cpcs])


    output = pl.DataFrame(results, columns)
    output.write_parquet(result_dataframe_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finds all potential matches to K in matching zone, aka set M.")
    parser.add_argument(
        "--raster",
        type=str,
        required=True,
        dest="m_raster_filename",
        help="GeoTIFF file containing pixels in set M as generated by build_m_raster.py"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching_zone_filename",
        help="Filename of GeoJSON file desribing area from which matching pixels may be selected."
    )
    parser.add_argument(
        "--lagged",
        type=str,
        required=True,
        dest="lagged",
        help="Boolean variable determining whether time-lagged matching will be used."
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
        help="Year of project evaluation"
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
        help="Directory containing Coarsened Proportional Coverage GeoTIFF tiles for all years."
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
        "--countries-raster",
        type=str,
        required=True,
        dest="countries_raster_filename",
        help="Raster of country IDs."
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
    args.lagged = args.lagged.lower() == "true"

    build_m_table(
        args.m_raster_filename,
        args.lagged,
        args.start_year,
        args.evaluation_year,
        args.matching_zone_filename,
        args.jrc_directory_path,
        args.cpc_directory_path,
        args.ecoregions_directory_path,
        args.elevation_directory_path,
        args.slope_directory_path,
        args.access_directory_path,
        args.countries_raster_filename,
        args.output_filename,
        args.processes_count
    )

if __name__ == "__main__":
    main()
