import glob
import os
import sys
import random
import math

import shapely # type: ignore
import numpy as np
import pandas
from geopandas import gpd # type: ignore
from yirgacheffe.layers import RasterLayer, VectorLayer, YirgacheffeLayer  # type: ignore
from yirgacheffe.window import PixelScale, Area  # type: ignore
from functools import partial
from methods.common.config import from_file

# TODO: Use Michael's code once available, for now this randomly selects pixels
# within a given area.
def find_matches(project : Area, area : Area, output : str, start_year : int, end_year : int, matches : int):
	# TODO: Decide format!!
	with open(output, "w") as file:
		file.write("src_lat,src_lng,match_lat,match_lng,")
		for year in range(start_year, end_year):
			file.write(f"luc_{year},")
		file.write("\n")
		for i in range(0, 10000):
			plng = random.uniform(project.left, project.right)
			plat = random.uniform(project.bottom, project.top)
			for j in range(0, matches):
				mlng = random.uniform(area.left, area.right)
				mlat = random.uniform(area.bottom, area.top)
				file.write(f"{plat},{plng},{mlat},{mlng},")
				for year in range(start_year, end_year):
					luc : int = math.floor(random.uniform(1.0, 7.0))
					file.write(f"{luc},")
				file.write("\n")

def generate_additionality_for_year(
	project_geojson_file : str,
	config_file : str,
	year: int,
	luc_raster_file : str,
	carbon_density : str,
	matches
) -> int:
	config = from_file(config_file)

	# Land use classes
	lucs = RasterLayer.layer_from_file(luc_raster_file)
	project_boundary = VectorLayer.layer_from_file(project_geojson_file, None, lucs.pixel_scale, lucs.projection)

	# LUCs only in project boundary
	intersection = RasterLayer.find_intersection([lucs, project_boundary])
	project_boundary.set_window_for_intersection(intersection)

	total_pixels = project_boundary.sum()
	lucs.set_window_for_intersection(intersection)

	def is_in_class(class_, data):
		return np.where(data != class_, 0.0, 1.0)

	lucs_in_project = lucs * project_boundary

	undisturbed = lucs_in_project.numpy_apply(partial(is_in_class, 1))
	degraded = lucs_in_project.numpy_apply(partial(is_in_class, 2))
	deforested = lucs_in_project.numpy_apply(partial(is_in_class, 3))
	regrowth = lucs_in_project.numpy_apply(partial(is_in_class, 4))
	water = lucs_in_project.numpy_apply(partial(is_in_class, 5))
	other = lucs_in_project.numpy_apply(partial(is_in_class, 6))

	proportions = np.array([
		undisturbed.sum() / total_pixels,
		degraded.sum() / total_pixels,
		deforested.sum() / total_pixels,
		regrowth.sum() / total_pixels,
		water.sum() / total_pixels,
		other.sum() / total_pixels
	])

	# Quick Sanity Check
	assert(np.sum(proportions) > 0.99 and np.sum(proportions) < 1.01)

	# TODO: the assumption of 30 x 30 resolution is not best practice
	areas = proportions * (total_pixels * 30 * 30)

	# TODO: may be present in config, in which case use that
	density_df = pandas.read_csv(carbon_density)
	density = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	# Density may have left some LUCs out like water
	for _, row in density_df.iterrows():
		luc = row["land use class"]
		density[int(luc) - 1] = row["carbon density"]

	# Total carbon densities per class
	s = areas * density


	# TODO: Check the math, I'm just hoping this all comes out in the wash
	# because we're using proportions so don't need to do each i individually,
	# this will become clearer when writing up the method.
	total_pixels_c = len(matches)

	counts_c = matches.value_counts(f"luc_{year}").sort_index().tolist()

	undisturbed_c = counts_c[0]
	degraded_c = counts_c[1]
	deforested_c = counts_c[2]
	regrowth_c = counts_c[3]
	water_c = counts_c[4]
	other_c = counts_c[5]

	proportions_c = np.array([
		undisturbed_c / total_pixels_c,
		degraded_c / total_pixels_c,
		deforested_c / total_pixels_c,
		regrowth_c / total_pixels_c,
		water_c / total_pixels_c,
		other_c / total_pixels_c
	])

	areas_c = proportions_c * (total_pixels * 30 * 30)
	s_c = areas_c * density

	return (s.sum() - s_c.sum())


if __name__ == "__main__":
	try:
		project_boundary_file = sys.argv[1]
		luc_file = sys.argv[2]
		project_config = sys.argv[3]
		carbon_density = sys.argv[4]
	except IndexError:
		print(f"Usage: {sys.argv[0]} PROJECT_BOUNDARY PROJECT_LUC PROJECT_CONFIG PROJECT_CARBON_DENSITY_CSV", file=sys.stderr)
		sys.exit(1)

	try:
		# TODO: 100 counterfactuals is an arbitrary amount to proxy a "convergence"
		filename = f"matches.csv"

		# TODO: Grab from configs
		start_year = 2012
		end_year = 2021

		# Just to have some slightly more realistic fake data
		# Land use classes
		lucs = RasterLayer.layer_from_file(luc_file)
		project_boundary = VectorLayer.layer_from_file(project_boundary_file, None, lucs.pixel_scale, lucs.projection)

		# LUCs only in project boundary
		intersection = RasterLayer.find_intersection([lucs, project_boundary])

		find_matches(intersection, intersection, filename, start_year, end_year, 100)
		matches = pandas.read_csv(filename)

		add = generate_additionality_for_year(
			project_boundary_file,
			project_config,
			2019,
			luc_file,
			carbon_density,
			matches
		)

		print(add)

	except FileNotFoundError as e:
		print(f"Failed to find file {e.filename}: {e.strerror}", file=sys.stderr)
		sys.exit(1)
