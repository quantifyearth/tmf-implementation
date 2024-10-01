import argparse
import datetime as dt
import json
import os
import tempfile
import geopandas as gpd  # type: ignore
import pandas as pd
import dotenv
from biomassrecovery.data.gedi_cmr_query import query  # type: ignore
from biomassrecovery.data.gedi_download_pipeline import check_and_format_shape  # type: ignore
from biomassrecovery.constants import GediProduct  # type: ignore
from osgeo import ogr, osr  # type: ignore

# load environment variables from .env file for Earthdata credentials
dotenv.load_dotenv()
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")


# function to chunk large geometries into smaller pieces for easier processing
def chunk_geometry(source: ogr.Layer, max_degrees: float) -> ogr.DataSource:
    # set up the spatial reference (coordinate system) for the output
    output_spatial_ref = osr.SpatialReference()
    output_spatial_ref.ImportFromEPSG(4326)  # WGS84 (standard for geographic coordinates)

    # create an in-memory layer to store the chunked geometries
    destination_data_source = ogr.GetDriverByName('Memory').CreateDataSource('in_memory')
    working_layer = destination_data_source.CreateLayer(
        "buffer", output_spatial_ref, geom_type=ogr.wkbMultiPolygon
    )
    feature_definition = working_layer.GetLayerDefn()

    # process each feature (geometry) from the source layer
    input_feature = source.GetNextFeature()
    while input_feature:
        geometry = input_feature.GetGeometryRef()
        min_lng, max_lng, min_lat, max_lat = geometry.GetEnvelope()

        origin_lat = min_lat
        # loop over latitude and longitude to create smaller polygon chunks
        while origin_lat < max_lat:
            far_lat = origin_lat + max_degrees

            origin_lng = min_lng
            while origin_lng < max_lng:
                far_lng = origin_lng + max_degrees

                # create a new chunk based on the bounding box of coordinates
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

                # only keep chunks that overlap with the original geometry
                if not intersection.IsEmpty():
                    new_feature = ogr.Feature(feature_definition)
                    new_feature.SetGeometry(intersection)
                    working_layer.CreateFeature(new_feature)

                origin_lng = far_lng
            origin_lat = far_lat
        input_feature = source.GetNextFeature()

    return destination_data_source


# function to fetch GEDI data based on the boundary file and save it to directories
def gedi_fetch(
    boundary_file: str,
    gedi_data_dir: str,
    output_csv: str,
    start_year: int,
    end_year: int
) -> None:
    # open the boundary file (GeoJSON, shapefile, etc.)
    boundary_dataset = ogr.Open(boundary_file)
    if boundary_dataset is None:
        raise ValueError("Failed to open boundary file")

    # create directories if they don't already exist
    output_dir = os.path.dirname(output_csv)
    os.makedirs(gedi_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    boundary_layer = boundary_dataset.GetLayer()

    # chunk the boundary geometries for easier processing
    chunked_dataset = chunk_geometry(boundary_layer, 0.4)
    chunked_layer = chunked_dataset.GetLayer()

    granule_metadatas = []
    chunk = chunked_layer.GetNextFeature()

    # use a temporary directory to store intermediary GeoJSON files
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = os.path.join(tmpdir, 'chunk.geojson')
        while chunk:
            # get the geometry from the chunk and write it to a GeoJSON file
            geometry = chunk.GetGeometryRef()

            # set up the output spatial reference for GeoJSON
            output_spatial_ref = osr.SpatialReference()
            output_spatial_ref.ImportFromEPSG(4326)  # WGS84 (global coordinate system)
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
            del destination_data_source  # close the GeoJSON file

            # read the GeoJSON as a GeoPandas dataframe for further processing
            shape = gpd.read_file(chunk_path)

            # simplify the shape to avoid excessive precision issues in coordinates
            shape = check_and_format_shape(shape, simplify=True, max_coords=200)

            # query the GEDI data for this shape and specified time range
            result = query(
                product=GediProduct.L4A,
                date_range=(dt.datetime(start_year, 1, 1), dt.datetime(end_year, 1, 1)),
                spatial=shape
            )
            granule_metadatas.append(result)

            # move to the next chunk
            chunk = chunked_layer.GetNextFeature()

    # concatenate all metadata into a single dataframe and remove duplicates
    granule_metadata = pd.concat(granule_metadatas).drop_duplicates(subset="granule_name")

    # write the granule names to a .csv file as the output
    granule_metadata.to_csv(output_csv, index=False)

    # save each granule's metadata as a JSON file in the specified GEDI data directory
    for _, row in granule_metadata.iterrows():
        basename, _ = os.path.splitext(row.granule_name)
        with open(os.path.join(gedi_data_dir, f"{basename}.json"), "w", encoding='utf-8') as f:
            json.dump(
                {
                    "name": row.granule_name,
                    "url": row.granule_url
                },
                f
            )


# main function to handle input arguments and call the data-fetching function
def main() -> None:
    # set up argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch GEDI data for the given boundary and save metadata as JSON and text files."
    )

    # add argument for the GEDI data directory where JSONs will be stored
    parser.add_argument(
        "--granules",
        type=str,
        required=True,
        dest="gedi_dir",
        help="Directory to store the GEDI JSON data files"
    )

    # add argument for the boundary file (e.g., GeoJSON, shapefile)
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        help="Path to the buffer boundary file"
    )

    # add argument for the output folder to store the text file with granule names
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .csv file with granule names will be saved"
    )

    args = parser.parse_args()  # parse the arguments from the command line

    # fetch GEDI data based on input parameters
    gedi_fetch(
        boundary_file=args.buffer,
        gedi_data_dir=args.gedi_dir,
        output_csv=args.output,
        start_year=2020,
        end_year=2021,
    )


if __name__ == "__main__":
    # run the main function if this script is executed
    main()
