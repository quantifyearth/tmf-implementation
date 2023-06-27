import json
import math
from typing import List, Optional

import numpy as np
import scipy
import shapely
from geopandas import gpd
from osgeo import ogr, osr # type: ignore

def utm_for_geometry(lat: float, lng: float) -> int:
	utm_band = (math.floor((lng + 180.0) / 6.0) % 60) + 1
	if lat < 0.0:
		epsg_code = 32700 + utm_band
	else:
		epsg_code = 32600 + utm_band
	return epsg_code

def expand_boundaries(source: gpd.GeoDataFrame, radius: float) -> gpd.GeoDataFrame:
	original_projection = source.crs

	utm_codes = np.array([utm_for_geometry(x.centroid.y, x.centroid.x) for x in source.geometry])
	utm_code = scipy.stats.mode(utm_codes, keepdims=False).mode
	projected_boundaries = source.to_crs(f"EPSG:{utm_code}")

	expanded = [x.buffer(radius) for x in projected_boundaries.geometry]
	simplified = shapely.unary_union(expanded)

	finished = gpd.GeoSeries(simplified)
	utm_result = gpd.GeoDataFrame.from_features(finished, crs=f"EPSG:{utm_code}")
	return utm_result.to_crs(original_projection)



def find_overlapping_geometries(src: ogr.Layer, guard: ogr.Layer) -> ogr.DataSource:
	"""Find geometries in src that overlap with the guard geometry"""
	output_spatial_ref = osr.SpatialReference()
	output_spatial_ref.ImportFromEPSG(4326) # aka WSG84

	destination_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
	working_layer = destination_data_source.CreateLayer("buffer", output_spatial_ref, geom_type=ogr.wkbMultiPolygon)
	feature_definition = working_layer.GetLayerDefn()

	if len(guard) == 0:
		return destination_data_source
	if len(src) == 0:
		return destination_data_source

	# work out guard envelope, and then filter input spatially on that
	# In practice, whilst this does speed up our loop, the time saved
	# is equally lost again in the call to src.SetSpatialFilter.
	#
	# I think long term there's an argument for pre-filtering the ecoregion
	# data to just regions that overlap any projects to speed this per project
	# calculation up.
	minX = 180.0
	minY = 90.0
	maxX = -180.0
	maxY = -90.0
	guard.ResetReading()
	guard_feature = guard.GetNextFeature()
	while guard_feature:
		guard_geometry = guard_feature.GetGeometryRef()
		envelope = guard_geometry.GetEnvelope()
		minX = min(minX, envelope[0])
		maxX = max(maxX, envelope[1])
		minY = min(minY, envelope[2])
		maxY = max(maxY, envelope[3])
		guard_feature = guard.GetNextFeature()
	frame = {
		'type': 'POLYGON',
		'coordinates': [
			[
				[minX, minY],
				[maxX, minY],
				[maxX, maxY],
				[minX, maxY],
				[minX, minY],
			]
		]
	}
	filter = ogr.CreateGeometryFromJson(json.dumps(frame))
	src.SetSpatialFilter(filter)

	src_feature = src.GetNextFeature()
	while src_feature:
		src_geometry = src_feature.GetGeometryRef()

		guard.ResetReading()
		guard_feature = guard.GetNextFeature()
		while guard_feature:
			guard_geometry = guard_feature.GetGeometryRef()
			if guard_geometry.Intersects(src_geometry):
				new_feature = ogr.Feature(feature_definition)
				new_feature.SetGeometry(src_geometry)
				working_layer.CreateFeature(new_feature)
				break

			guard_feature = guard.GetNextFeature()

		src_feature = src.GetNextFeature()
	return destination_data_source

def intersect_all_layers(
	layers: List[ogr.Layer],
	destination_data_source: Optional[ogr.DataSource]
) -> ogr.DataSource:
	if destination_data_source is None:
		destination_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
	working_layer = destination_data_source.CreateLayer("intersection", output_spatial_ref, geom_type=ogr.wkbMultiPolygon)

