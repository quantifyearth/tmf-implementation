import json
import math
from typing import List, Optional

from osgeo import ogr, osr # type: ignore

def utm_for_geometry(geometry: ogr.Geometry) -> int:
	# work out centre of geometry
	envelope = geometry.GetEnvelope()
	centre_x = envelope[0] + ((envelope[1] - envelope[0]) / 2)
	centre_y = envelope[2] + ((envelope[3] - envelope[2]) / 2)

	utm_band = (math.floor((centre_x + 180.0) / 6.0) % 60) + 1
	if centre_y < 0.0:
		epsg_code = 32700 + utm_band
	else:
		epsg_code = 32600 + utm_band

	return epsg_code

def expand_polygon(
	geometry: ogr.Geometry,
	input_spatial_ref: osr.SpatialReference,
	output_spatial_ref: osr.SpatialReference,
	distance_in_metres: int,
) -> ogr.Geometry:
	if geometry.GetGeometryType() != ogr.wkbPolygon:
		raise ValueError("Can only expand polygons")

	utmband = utm_for_geometry(geometry)

	working_spatial_ref = osr.SpatialReference()
	working_spatial_ref.ImportFromEPSG(utmband)
	coordTransTo = osr.CoordinateTransformation(input_spatial_ref, working_spatial_ref)
	coordTransFrom = osr.CoordinateTransformation(working_spatial_ref, output_spatial_ref)

	geometry.Transform(coordTransTo)
	expanded = geometry.Buffer(distance_in_metres)
	expanded.Transform(coordTransFrom)

	return expanded

# Convert a set of polygons to have a boundary that is uniform in all
# directions by converting to UTM first.
#
# The type signature here is weird due to SWIG weirdness
# You have to select the layers from the source dataset by apply
# filters first, so the input has to be the layer of the source
# dataset. However, Layers are created in the context of a
# destination dataset, and I've had issues in the past where
# if I just return the layers, the dataset is GC'd even though
# we never care about it.
def expand_boundaries(source: ogr.Layer, distance_in_metres: int, destination_data_source: Optional[ogr.DataSource]=None) -> ogr.DataSource:
	input_spatial_ref = osr.SpatialReference()
	input_spatial_ref.ImportFromEPSG(4326) # aka WSG84
	output_spatial_ref = input_spatial_ref # just to make the code read less weird

	if destination_data_source is None:
		destination_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
	working_layer = destination_data_source.CreateLayer("buffer", output_spatial_ref, geom_type=ogr.wkbMultiPolygon)

	expanded_geometries = ogr.Geometry(ogr.wkbMultiPolygon)
	input_feature = source.GetNextFeature()
	while input_feature:
		geometry = input_feature.GetGeometryRef()
		match geometry.GetGeometryType():
			case ogr.wkbPolygon:
				expanded = expand_polygon(geometry, input_spatial_ref, output_spatial_ref, distance_in_metres)
				res = expanded_geometries.AddGeometry(expanded)
				if res != ogr.OGRERR_NONE:
					raise Exception("Failed to add geometry")

			case ogr.wkbMultiPolygon:
				for i in range(geometry.GetGeometryCount()):
					child_polygon = geometry.GetGeometryRef(i)
					expanded = expand_polygon(child_polygon, input_spatial_ref, output_spatial_ref, distance_in_metres)
					res = expanded_geometries.AddGeometry(expanded)
					if res != ogr.OGRERR_NONE:
						raise Exception("Failed to add geometry")

			case ogr.wkbGeometryCollection:
				raise NotImplementedError("not yet implemented for geometry collection")

			case other:
				# skip for lines, points, etc.
				pass

		input_feature = source.GetNextFeature()

	minimal_geometries = expanded_geometries.UnionCascaded()

	feature_definition = working_layer.GetLayerDefn()
	new_feature = ogr.Feature(feature_definition)
	new_feature.SetGeometry(minimal_geometries)
	working_layer.CreateFeature(new_feature)

	return destination_data_source

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

