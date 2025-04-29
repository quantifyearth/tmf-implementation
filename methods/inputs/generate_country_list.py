import argparse
import json

from geopandas import gpd  # type: ignore

def generate_country_list(
    buffer_boundary_filename: str,
    countries_vector_filename: str,
    output_filename: str
) -> None:
    buffer_boundaries = gpd.read_file(buffer_boundary_filename)
    countries = gpd.read_file(countries_vector_filename)
    matches = countries.sjoin(buffer_boundaries)
    with open(output_filename, "w", encoding="utf-8") as outfd:
        outfd.write(json.dumps(list(set(matches['ISO_A2']))))

def main() -> None:
    parser = argparse.ArgumentParser(description="Finds the country codes for those the project intersects with")
    parser.add_argument(
        "--buffer",
        type=str,
        required=True,
        dest="buffer_boundary_filename",
        help="GeoJSON File of buffer boundary."
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
        help="JSON file listing country codes."
    )
    args = parser.parse_args()

    generate_country_list(
        args.buffer_boundary_filename,
        args.countries_vector_filename,
        args.output_filename,
    )

if __name__ == "__main__":
    main()
