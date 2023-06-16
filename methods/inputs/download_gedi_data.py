import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from typing import Set

from biomassrecovery import constants # type: ignore
import geopandas as gpd # type: ignore
import pandas as pd
import requests
from biomassrecovery.data.gedi_cmr_query import query  # type: ignore
from biomassrecovery.spark.gedi_download_pipeline import _check_and_format_shape, _query_downloaded, _granules_table # type: ignore
from biomassrecovery.constants import GediProduct  # type: ignore
from osgeo import ogr, osr  # type: ignore

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
				chunk_geometry = ogr.CreateGeometryFromJson(json.dumps(frame))
				intersection = geometry.Intersection(chunk_geometry)
				if not intersection.IsEmpty():
					new_feature = ogr.Feature(feature_definition)
					new_feature.SetGeometry(intersection)
					working_layer.CreateFeature(new_feature)

				origin_lng = far_lng
			origin_lat = far_lat
		input_feature = source.GetNextFeature()

	return destination_data_source

def download_granule(gedi_data_dir: str, name: str, url: str) -> None:
	with tempfile.TemporaryDirectory() as tmpdir:
		with requests.Session() as session:
			session.auth = (constants.EARTHDATA_USER, constants.EARTHDATA_PASSWORD)
			response = session.get(url, stream=True)
			download_target_name = os.path.join(tmpdir, name)
			with open(download_target_name, 'wb') as f:
				for chunk in response.iter_content(chunk_size=1024*1024):
					f.write(chunk)

		final_name = os.path.join(gedi_data_dir, name)
		shutil.move(download_target_name, final_name)

def gedi_fetch(boundary_file: str, gedi_data_dir: str) -> None:
	boundary_dataset = ogr.Open(boundary_file)
	if boundary_dataset is None:
		raise ValueError("Failed top open boundry file")
	os.makedirs(gedi_data_dir, exist_ok=True)

	boundary_layer = boundary_dataset.GetLayer()
	chunked_dataset = chunk_geometry(boundary_layer, 0.5)
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
			working_layer = destination_data_source.CreateLayer("buffer", output_spatial_ref, geom_type=ogr.wkbMultiPolygon)
			feature_definition = working_layer.GetLayerDefn()
			new_feature = ogr.Feature(feature_definition)
			new_feature.SetGeometry(geometry)
			working_layer.CreateFeature(new_feature)
			del destination_data_source # aka destination_data_source.Close()

			shape = gpd.read_file(chunk_path)
			shape = _check_and_format_shape(shape)
			result = query(
				product=GediProduct.L4A,
				date_range=(dt.datetime(2020, 1, 1, 0, 0), dt.datetime(2021, 1, 1, 0, 0)),
				spatial=shape
			)
			granule_metadatas.append(result)

			chunk = chunked_layer.GetNextFeature()

	granule_metadata = pd.concat(granule_metadatas).drop_duplicates(subset="granule_name")

	stored_granules = [file for file in os.listdir(gedi_data_dir) if file.endswith('.h5')]
	required_granules = granule_metadata.loc[~granule_metadata["granule_name"].isin(stored_granules)]
	name_url_pairs = required_granules[["granule_name", "granule_url"]].to_records(
		index=False
	)

	for name, url in name_url_pairs:
		download_granule(gedi_data_dir, name, url)

if __name__ == "__main__":
	try:
		boundary_file = sys.argv[1]
		gedi_data_dir = sys.argv[2]
	except IndexError:
		print(f"Usage: {sys.argv[0]} BUFFER_BOUNDRY_FILE GEDI_DATA_DIRECTORY")
		sys.exit(1)

	gedi_fetch(boundary_file, gedi_data_dir)