import datetime as dt
import json
import os
import sys
import tempfile

import geopandas as gpd # type: ignore
import pandas as pd
import dotenv
from biomassrecovery.data.gedi_cmr_query import query  # type: ignore
from biomassrecovery.data.gedi_download_pipeline import check_and_format_shape # type: ignore
from biomassrecovery.constants import GediProduct  # type: ignore
from osgeo import ogr, osr  # type: ignore

from methods.common import DownloadError

# This is defined in biomassrecovery.environment too, but that file
# is full of side-effects, so just import directly here.
dotenv.load_dotenv()
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")


def chunk_geometry(source: ogr.Layer, max_degrees: float) -> ogr.DataSource:
    output_spatial_ref = osr.SpatialReference()
    output_spatial_ref.ImportFromEPSG(4326) # aka WSG84

    destination_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
    working_layer = destination_data_source.CreateLayer("buffer", output_spatial_ref, geom_type=ogr.wkbMultiPolygon)
    feature_definition = working_layer.GetLayerDefn()

    input_feature = source.GetNextFeature()
    while input_feature:
        geometry = input_feature.GetGeometryRef()
        min_lng, max_lng, min_lat, max_lat = geometry.GetEnvelope()

        origin_lat = min_lat
        while origin_lat < max_lat:
            far_lat = origin_lat + max_degrees

            origin_lng = min_lng
            while origin_lng < max_lng:
                far_lng = origin_lng + max_degrees

                frame = {
                    'type': 'POLYGON',
                    'coordinates': [
                        [
                            [origin_lng, origin_lat],
                            [far_lng,    origin_lat],
                            [far_lng,    far_lat],
                            [origin_lng, far_lat],
                            [origin_lng, origin_lat],
                        ]
                    ]
                }
                chunk = ogr.CreateGeometryFromJson(json.dumps(frame))
                intersection = geometry.Intersection(chunk)
                if not intersection.IsEmpty():
                    new_feature = ogr.Feature(feature_definition)
                    new_feature.SetGeometry(intersection)
                    working_layer.CreateFeature(new_feature)

                origin_lng = far_lng
            origin_lat = far_lat
        input_feature = source.GetNextFeature()

    return destination_data_source

def gedi_fetch(boundary_file: str, gedi_data_dir: str) -> None:
    boundary_dataset = ogr.Open(boundary_file)
    if boundary_dataset is None:
        raise ValueError("Failed top open boundry file")
    os.makedirs(gedi_data_dir, exist_ok=True)

    boundary_layer = boundary_dataset.GetLayer()
    chunked_dataset = chunk_geometry(boundary_layer, 0.4)
    chunked_layer = chunked_dataset.GetLayer()

    granule_metadatas = []
    chunk = chunked_layer.GetNextFeature()
    # I had wanted to use the in-memory file system[0] that allegedly both GDAL and Geopandas support to do
    # this, but for some reason despite seeing other use it in examples[1], it isn't working for me, and so
    # I'm bouncing everything via the file system ¯\_(ツ)_/¯
    # [0] https://gdal.org/user/virtual_file_systems.html#vsimem-in-memory-files
    # [1] https://gis.stackexchange.com/questions/440820/loading-a-ogr-data-object-into-a-geodataframe
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = os.path.join(tmpdir, 'chunk.geojson')
        while chunk:
            # The biomassrecovery code works in geopanddas shapes, so we need to
            # cover this feature to a geopandas object
            geometry = chunk.GetGeometryRef()

            output_spatial_ref = osr.SpatialReference()
            output_spatial_ref.ImportFromEPSG(4326) # aka WSG84
            destination_data_source = ogr.GetDriverByName('GeoJSON').CreateDataSource(chunk_path)
            working_layer = destination_data_source.CreateLayer(
                "buffer",
                output_spatial_ref,
                geom_type=ogr.wkbMultiPolygon
            )
            feature_definition = working_layer.GetLayerDefn()
            new_feature = ogr.Feature(feature_definition)
            new_feature.SetGeometry(geometry)
            working_layer.CreateFeature(new_feature)
            del destination_data_source # aka destination_data_source.Close()

            shape = gpd.read_file(chunk_path)

            # We set max_coords rather low (the upper bound is less than 5000) as some of
            # our geometries are causing issues with being too precise and the coordinates
            # not all being > 1m apart. Checking this for every geometry is a bit hard, so
            # instead we approximate it by setting the bar for simplification much lower.
            shape = check_and_format_shape(shape, simplify=True, max_coords=200)
            result = query(
                product=GediProduct.L4A,
                date_range=(dt.datetime(2020, 1, 1, 0, 0), dt.datetime(2021, 1, 1, 0, 0)),
                spatial=shape
            )
            granule_metadatas.append(result)

            chunk = chunked_layer.GetNextFeature()

    granule_metadata = pd.concat(granule_metadatas).drop_duplicates(subset="granule_name")

    for _, row in granule_metadata.iterrows():
        basename, _ = os.path.splitext(row.granule_name)
        with open(os.path.join(gedi_data_dir, f"{basename}.json"), "w", encoding='utf-8') as f:
            f.write(json.dumps(
                {
                    "name": row.granule_name,
                    "url": row.granule_url
                }
            ))

def main() -> None:
    try:
        boundary_file = sys.argv[1]
        gedi_data_dir = sys.argv[2]
    except IndexError:
        print(f"Usage: {sys.argv[0]} BUFFER_BOUNDRY_FILE GEDI_DATA_DIRECTORY")
        sys.exit(1)
    except DownloadError as exc:
        print(f"Failed to download: {exc.msg}")
        sys.exit(1)

    gedi_fetch(boundary_file, gedi_data_dir)

if __name__ == "__main__":
    main()
