import math
import os
import sys
from typing import Optional

from osgeo import ogr, osr # type: ignore

PROJECT_BOUNDARY_RADIUS_IN_METRES = 30000

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

	source.ResetReading()
	input_feature = source.GetNextFeature()
	while input_feature:
		geometry = input_feature.GetGeometryRef()
		# we do this per geometry as a project may cross UTM bounds
		utmband = utm_for_geometry(geometry)

		working_spatial_ref = osr.SpatialReference()
		working_spatial_ref.ImportFromEPSG(utmband)
		coordTransTo = osr.CoordinateTransformation(input_spatial_ref, working_spatial_ref)
		coordTransFrom = osr.CoordinateTransformation(working_spatial_ref, output_spatial_ref)

		geometry.Transform(coordTransTo)
		expanded = geometry.Buffer(distance_in_metres)
		expanded.Transform(coordTransFrom)

		feature_definition = working_layer.GetLayerDefn()
		new_feature = ogr.Feature(feature_definition)
		new_feature.SetGeometry(expanded)
		working_layer.CreateFeature(new_feature)

		input_feature = source.GetNextFeature()

	return destination_data_source

def generate_boundary(input_filename: str, output_filename: str, filter: Optional[str]) -> None:
	project_boundaries = ogr.Open(input_filename)
	if project_boundaries is None:
		print(f"Failed to open {input_filename}", file=sys.stderr)
		sys.exit(1)

	if filter is not None:
		project_boundaries.SetAttributeFilter(filter)
	project_layer = project_boundaries.GetLayer()

	_, ext = os.path.splitext(output_filename)
	driver_name = None
	match ext.lower():
		case '.geojson':
			driver_name = 'GeoJSON'
		case '.gpkg':
			driver_name = 'GPKG'
		case '.shp':
			driver_name = 'ESRP Shapefile'
		case other:
			print(f"{ext} is not a supported file type", file=sys.stderr)
			sys.exit(1)

	target_dataset = ogr.GetDriverByName(driver_name).CreateDataSource(output_filename)
	_ = expand_boundaries(project_layer, PROJECT_BOUNDARY_RADIUS_IN_METRES, target_dataset)

if __name__ == "__main__":
	try:
		input_filename = sys.argv[1]
		output_filename = sys.argv[2]
	except IndexError:
		print(f"Usage: {sys.argv[0]} INPUT OUTPUT [filter]", file=sys.stderr)
		sys.exit(1)

	filter: Optional[str] = None
	try:
		filter = sys.argv[3]
	except IndexError:
		pass

	generate_boundary(input_filename, output_filename, filter)
