import glob
import os
import sys
from collections import namedtuple

import pandas as pd
from geopandas import gpd  # type: ignore
from yirgacheffe.layers import GroupLayer, RasterLayer, VectorLayer, YirgacheffeLayer  # type: ignore
from yirgacheffe.window import PixelScale  # type: ignore

from methods.common.geometry import area_for_geometry

HECTARE_WIDTH_IN_METERS = 100
PIXEL_WIDTH_IN_METERS = 30
SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.25
LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE = 0.05

# The '2 *' in this is because I'm just considering one axis, rather than area
PIXEL_SKIP_SMALL_PROJECT = round((HECTARE_WIDTH_IN_METERS / (2 * SMALL_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)
PIXEL_SKIP_LARGE_PROJECT = round((HECTARE_WIDTH_IN_METERS / (2 * LARGE_PROJECT_PIXEL_DENSITY_PER_HECTARE)) / PIXEL_WIDTH_IN_METERS)

MatchingCollection = namedtuple('MatchingCollection', ['boundary', 'lucs', 'ecoregions', 'elevation'])

def build_layer_collection(
	pixel_scale: PixelScale,
	projection: str,
	evaluation_year: int,
	boundary_filename: str,
	jrc_data_folder: str,
	ecoregions_filename: str,
	elevation_folder_filename: str,
) -> MatchingCollection:
	outline_layer = VectorLayer.layer_from_file(boundary_filename, None, pixel_scale, projection)
	lucs = [
		GroupLayer([
			RasterLayer.layer_from_file(os.path.join(jrc_data_folder, filename)) for filename in
				glob.glob(f"*{evaluation_year - offset}*.tif", root_dir=jrc_data_folder)
		]) for offset in range(0, 11, 5)
	]

	# ecoregions is such a heavy layer it pays to just rasterize it once - we should possibly do this once
	# as part of import of the ecoregions data
	ecoregions = VectorLayer.layer_from_file(ecoregions_filename, None, pixel_scale, projection, burn_value="ECO_ID")
	ecoregions.set_window_for_intersection(outline_layer.area)
	rastered_ecoregions = RasterLayer.empty_raster_layer_like(ecoregions)
	ecoregions.save(rastered_ecoregions)

	elevation = GroupLayer([
		RasterLayer.layer_from_file(os.path.join(elevation_folder_filename, filename)) for filename in
			glob.glob(f"*.tif", root_dir=elevation_folder_filename)
	])

	# constrain everything to project boundaries
	layers = [elevation] + lucs
	for layer in layers:
		layer.set_window_for_intersection(outline_layer.area)

	return MatchingCollection(
		boundary=outline_layer,
		lucs=lucs,
		ecoregions=rastered_ecoregions,
		elevation=elevation
	)

def calculate_k(
	project_boundary_filename: str,
	jrc_data_folder: str,
	ecoregions_filename: str,
	elevation_folder_filename: str,
	year: int
) -> None:

	project = gpd.read_file(project_boundary_filename)
	project_area_in_metres_squared = area_for_geometry(project)
	project_area_in_hectares = project_area_in_metres_squared / 10_000
	pixel_skip = PIXEL_SKIP_LARGE_PROJECT if (project_area_in_hectares > 250_000) else PIXEL_SKIP_SMALL_PROJECT

	# everything is done at JRC resolution, so load a sample file from there first to get the ideal pixel scale
	example_jrc_filename = glob.glob("*.tif", root_dir=jrc_data_folder)[0]
	example_jrc_layer = RasterLayer.layer_from_file(os.path.join(jrc_data_folder, example_jrc_filename))

	project_collection = build_layer_collection(
		example_jrc_layer.pixel_scale,
		example_jrc_layer.projection,
		year,
		project_boundary_filename,
		jrc_data_folder,
		ecoregions_filename,
		elevation_folder_filename,
	)

	results = []

	project_width = project_collection.boundary.window.xsize
	for yoffset in range(0, project_collection.boundary.window.ysize, pixel_skip):
		row_boundary = project_collection.boundary.read_array(0, yoffset, project_width, 1)
		row_elevation = project_collection.elevation.read_array(0, yoffset, project_width, 1)
		row_ecoregion = project_collection.ecoregions.read_array(0, yoffset, project_width, 1)
		row_luc = [
			luc.read_array(0, yoffset, project_width, 1) for luc in project_collection.lucs
		]
		for xoffset in range(0, project_width, pixel_skip):
			if not row_boundary[0][xoffset]:
				continue
			lucs = [x[0][xoffset] for x in row_luc]
			results.append([xoffset, yoffset, row_elevation[0][xoffset], row_ecoregion[0][xoffset]] + lucs)

	df = pd.DataFrame(results, columns=['x', 'y', 'elevation', 'ecoregion', 'luc0', 'luc5', 'luc10'])
	df.to_parquet(result_dataframe_filename)

if __name__ == "__main__":
	try:
		project_boundary_filename = sys.argv[1]
		jrc_data_folder = sys.argv[2]
		ecoregions_filename = sys.argv[3]
		elevation_folder_filename = sys.argv[4]
		year = int(sys.argv[5])
		result_dataframe_filename = sys.argv[6]
	except (IndexError, ValueError):
		print(f"Usage: {sys.argv[0]} PROJECT_BOUNDARY JRC_FOLDER ECOREGIONS_GEOJSON ELEVATION_FOLDER YEAR OUTPUT_PARQUET", file=sys.stderr)
		sys.exit(1)

	calculate_k(
		project_boundary_filename,
		jrc_data_folder,
		ecoregions_filename,
		elevation_folder_filename,
		year,
	)
