import argparse
import json
import os
import math
from typing import Dict, List

import geopandas as gpd  # type: ignore
import numpy as np
import pandas as pd
from biomassrecovery.data.gedi_database import GediDatabase, RegEx  # type: ignore
from osgeo import ogr  # type: ignore
from yirgacheffe.layers import RasterLayer  # type: ignore

def generate_carbon_density(boundary_file: str, luc_raster_file: str, output_file: str) -> None:

    _, output_ext = os.path.splitext(output_file)
    output_ext = output_ext.lower()
    if output_ext not in ['.csv', '.parquet']:
        raise ValueError("We only support .csv and .parquet outputs.")

    boundary_dataset = ogr.Open(boundary_file)
    if boundary_dataset is None:
        raise ValueError("Could not open boundary file")
    boundary_layer = boundary_dataset.GetLayer()

    jrc_raster_layer = RasterLayer.layer_from_file(luc_raster_file)
    gedi_db = GediDatabase()

    pixel_scale = jrc_raster_layer.pixel_scale
    luc_buckets: Dict[int, List[float]] = {}

    for yoffset in range(1, jrc_raster_layer.window.ysize - 2):

        # Create a geometry that represents one row of pixels in the LUC raster
        y_top = jrc_raster_layer.area.top + (yoffset * pixel_scale.ystep)
        y_bottom = y_top + pixel_scale.ystep
        frame = {
            'type': 'POLYGON',
            'coordinates': [
                [
                    [jrc_raster_layer.area.left,  y_top],
                    [jrc_raster_layer.area.right, y_top],
                    [jrc_raster_layer.area.right, y_bottom],
                    [jrc_raster_layer.area.left,  y_bottom],
                    [jrc_raster_layer.area.left,  y_top],
                ]
            ]
        }
        geometry_slice = ogr.CreateGeometryFromJson(json.dumps(frame))

        wkts = []
        boundary_layer.ResetReading()
        feature = boundary_layer.GetNextFeature()
        while feature:
            geometry = feature.GetGeometryRef()
            intersection = geometry.Intersection(geometry_slice)
            if not intersection.IsEmpty():
                wkts.append(intersection.ExportToWkt())
            feature = boundary_layer.GetNextFeature()
        if len(wkts) == 0:
            continue
        as_series = gpd.GeoSeries.from_wkt(wkts)

        # The date limits here come from discussion with Miranda Lam:
        # "we currently use the JRC Annual Change 2020 for land cover class and GEDI shots from 2020/1/1 to 2021/1/1"
        data = gedi_db.query(
            'level_4a',
            columns=['agbd'],
            start_time="2020/01/01",
            end_time="2021/01/01",
            geometry=as_series,
            order_by=["-absolute_time"],
            degrade_flag=0,
            beam_type="full",
            l4_quality_flag=1,
            # leaf_off_flag=0, - this doesn't work on the current live DB
            granule_name=RegEx(r".*02_V002\.h5"), # Filters out older GEDI data versions
        )
        for _, row in data.iterrows():
            agbd = row['agbd']
            point = row['geometry']

            xoffset = math.floor((point.x - jrc_raster_layer.area.left) / pixel_scale.xstep)

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
        "--buffer",
        type=str,
        required=True,
        dest="buffer_boundary_filename",
        help="Buffer boundary file"
    )
    parser.add_argument(
        "--luc",
        type=str,
        required=True,
        dest="luc_raster_file",
        help="File of country vector shapes."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Output file name."
    )
    args = parser.parse_args()

    boundary_file = args.buffer_boundary_filename
    luc_raster_file = args.luc_raster_file
    output_file = args.output_filename

    generate_carbon_density(boundary_file, luc_raster_file, output_file)

if __name__ == "__main__":
    main()
