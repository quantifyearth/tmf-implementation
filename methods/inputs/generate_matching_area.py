import errno
import sys

from osgeo import ogr

from methods.common.geometry import expand_polygon, find_overlapping_geometries, expand_boundaries

MATCHING_RADIUS_IN_METRES = 2_000_000


def generate_matching_area(
	project_shape_filename: str,
	country_iso_a2_code: str,
	countries_shape_filename: str,
	ecoregions_shape_filename: str,
	output_shape_filename: str
) -> None:
	project_boundaries = ogr.Open(project_shape_filename)
	if project_boundaries is None:
		raise FileNotFoundError(errno.ENOENT, "Project file not found", project_shape_filename)
	project_layer = project_boundaries.GetLayer()
	extended_project = expand_boundaries(project_layer, MATCHING_RADIUS_IN_METRES)

	country_shapes_datasource = ogr.Open(countries_shape_filename)
	if country_shapes_datasource is None:
		raise FileNotFoundError(errno.ENOENT, "Country shapes file not found", countries_shape_filename)
	country_shapes = country_shapes_datasource.GetLayer()
	if len(country_iso_a2_code) != 2:
		raise ValueError(f"Expected country code to be 2 letters long, got {country_iso_a2_code}")
	country_shapes.SetAttributeFilter(f"ISO_A2 = '{country_iso_a2_code}'")
	if len(country_shapes) == 0:
		raise ValueError(f"Got no matches for country ISO A2 code {country_iso_a2_code}")
	elif len(country_shapes) > 1:
		raise ValueError(f"Got too many matches for country ISO A2 code {country_iso_a2_code}")

	ecoregions = ogr.Open(ecoregions_shape_filename)
	if ecoregions is None:
		raise FileNotFoundError(errno.ENOENT, "Ecoregions file not found", ecoregions_shape_filename)
	overlapping_ecoregions = find_overlapping_geometries(ecoregions.GetLayer(), project_boundaries.GetLayer())

	# MISSING: Project REDD projects!

	# target is the intersection of these zones



if __name__ == "__main__":
	try:
		project_shape_filename = sys.argv[1]
		country_iso_a2_code = sys.argv[2]
		countries_shape_filename = sys.argv[3]
		ecoregions_shape_filename = sys.argv[4]
		output_shape_filename = sys.argv[5]
	except IndexError:
		print(f"Usage: {sys.argv[0]} PROJECT_SHAPEFILE COUNTRY_CODE COUNTRY_SHAPEFILE ECOREGIONS_SHAPEFILE OUTPUT_SHAPEFILE", file=sys.stderr)
		sys.exit(1)

	try:
		generate_matching_area(
			project_shape_filename,
			country_iso_a2_code,
			countries_shape_filename,
			ecoregions_shape_filename,
			output_shape_filename
		)
	except FileNotFoundError as exc:
		print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
		sys.exit(1)
	except ValueError as exc:
		printf(f"Bad value: {exc.msg}")
		sys.exit(1)

