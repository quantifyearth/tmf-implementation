import argparse
import json

from geopandas import gpd

def generate_country_list(
	leakage_boundary_filename: str,
	countries_vector_filename: str,
	output_filename: str
) -> None:
	leakage_boundaries = gpd.read_file(leakage_boundary_filename)
	countries = gpd.read_file(countries_vector_filename)
	matches = countries.sjoin(leakage_boundaries)
	with open(output_filename, 'w') as outfd:
		outfd.write(json.dumps(list(set(matches['ISO_A2']))))

def main() -> None:
	parser = argparse.ArgumentParser(description="Finds the country codes for those the project intersects with")
	parser.add_argument(
		"--leakage",
		type=str,
		required=True,
		dest="leakage_boundary_filename",
		help="GeoJSON File of leakage boundary."
	)
	parser.add_argument(
		"--countries",
		type=str,
		required=True,
		dest="countries_vector_filename",
		help="File of country vector shapes."
	)
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		dest="output_filename",
		help="JSON file listing contry codes."
	)
	args = parser.parse_args()

	generate_country_list(
		args.leakage_boundary_filename,
		args.countries_vector_filename,
		args.output_filename,
	)

if __name__ == "__main__":
	main()
