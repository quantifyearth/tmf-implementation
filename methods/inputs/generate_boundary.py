import errno
import math
import os
import sys
from typing import Optional

from osgeo import ogr, osr # type: ignore

from methods.common.geometry import expand_boundaries

PROJECT_BOUNDARY_RADIUS_IN_METRES = 30_000

def generate_boundary(input_filename: str, output_filename: str, filter: Optional[str]) -> None:
	project_boundaries = ogr.Open(input_filename)
	if project_boundaries is None:
		raise FileNotFoundError(errno.ENOENT, "Project file not found", project_shape_filename)

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

	try:
		generate_boundary(input_filename, output_filename, filter)
	except FileNotFoundError as e:
		print(f"Failed to find file {e.filename}: {e.strerror}", file=sys.stderr)
		sys.exit(1)

