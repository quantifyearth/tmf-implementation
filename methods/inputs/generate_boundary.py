import sys

from fiona.errors import DriverError
from geopandas import gpd

from methods.common.geometry import expand_boundaries

PROJECT_BOUNDARY_RADIUS_IN_METRES = 30_000

def generate_boundary(input_filename: str, output_filename: str) -> None:
	project_boundaries = gpd.read_file(input_filename)
	result = expand_boundaries(project_boundaries, PROJECT_BOUNDARY_RADIUS_IN_METRES)b
	result.to_file(output_filename, driver="GeoJSON")

if __name__ == "__main__":
	try:
		input_filename = sys.argv[1]
		output_filename = sys.argv[2]
	except IndexError:
		print(f"Usage: {sys.argv[0]} INPUT OUTPUT [filter]", file=sys.stderr)
		sys.exit(1)

	try:
		generate_boundary(input_filename, output_filename)
	except FileNotFoundError as e:
		print(f"Failed to find file {e.filename}: {e.strerror}", file=sys.stderr)
		sys.exit(1)
	except DriverError as exc:
		print(exc.args[0], file=sys.stderr)
		sys.exit(1)
