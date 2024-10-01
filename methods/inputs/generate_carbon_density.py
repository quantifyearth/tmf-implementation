import argparse
import glob
import os
import math
from typing import Dict, List
import glob

import geopandas as gpd  # type: ignore
import numpy as np
import pandas as pd
from yirgacheffe.layers import RasterLayer, GroupLayer  # type: ignore

def generate_carbon_density(
    jrc_directory_path: str,
    gedi_data_file: str,
    output_file: str
) -> None:

    _, output_ext = os.path.splitext(output_file)
    output_ext = output_ext.lower()
    if output_ext not in ['.csv', '.parquet']:
        raise ValueError("We only support .csv and .parquet outputs.")

    jrc_raster_layer = GroupLayer([
        RasterLayer.layer_from_file(os.path.join(jrc_directory_path, filename)) for filename in
            glob.glob("*2020*.tif", root_dir=jrc_directory_path)
    ], name="luc_2020")

    gedi = gpd.read_file(gedi_data_file)

    pixel_scale = jrc_raster_layer.pixel_scale
    luc_buckets: Dict[int, List[float]] = {}

    for _, row in gedi.iterrows():
        agbd = row['agbd']
        point = row['geometry']

        xoffset = math.floor((point.x - jrc_raster_layer.area.left) / pixel_scale.xstep)
        yoffset = math.floor((point.y - jrc_raster_layer.area.top) / pixel_scale.ystep)

        # Check the JRC data around this cell
        surroundings = jrc_raster_layer.read_array(xoffset - 1, yoffset - 1, 3, 3)
        land_use_class = surroundings[1][1]

        if not np.all(surroundings == land_use_class):
            continue

        try:
            luc_buckets[land_use_class].append(agbd)
        except KeyError:
            luc_buckets[land_use_class] = [agbd]

    land_use_class_list = list(luc_buckets.keys())
    land_use_class_list.sort()

    results = []
    for land_use_class in land_use_class_list:
        median_agbd = np.median(luc_buckets[land_use_class])
        bgbd = median_agbd * 0.2
        deadwood_bd = median_agbd * 0.11
        total = median_agbd + bgbd + deadwood_bd
        carbon_density = total * 0.47
        results.append([land_use_class, carbon_density])

    output = pd.DataFrame(results, columns=["land use class", "carbon density"])
    match output_ext:
        case '.csv':
            output.to_csv(output_file, index=False)
        case '.parquet':
            output.to_parquet(output_file)
        case _:
            assert False, "Extensions was validated earlier"

def main() -> None:
    parser = argparse.ArgumentParser(description="Finds the country codes for those the project intersects with")
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

    generate_carbon_density(
        args.jrc_directory_path,
        args.gedi_data_file,
        args.output_filename)

if __name__ == "__main__":
    main()
