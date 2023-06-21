import os
import sys
import tempfile

import geopandas # type: ignore
import requests

from biomassrecovery.utils.unzip import unzip  # type: ignore

# As taken from the World Bank data catalog https://datacatalog.worldbank.org/search/dataset/0038272
# Under CC Attribution 4.0
SOURCE_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0046659/wb_countries_admin0_10m.zip"

class DownloadError(Exception):
	"""Indicate the download failed"""

# The World Bank high detail country polygons are available as a zipped
# shape file. Thus we need to do the following:
#
# 1: download the zip filer
# 2: unzip that file
# 3: load the shape file
# 4: convert it to something more useful (i.e., geojson/gpkg)
def download_country_polygons(target_filename: str) -> None:
	with tempfile.TemporaryDirectory() as tmpdir:
		download_path = os.path.join(tmpdir, "countries.zip")
		response = requests.get(SOURCE_URL, stream=True)
		if not response.ok:
			raise DownloadError(f'{response.status_code}: {response.reason}')
		with open(download_path, 'wb') as fd:
			for chunk in response.iter_content(chunk_size=1024*1024):
				fd.write(chunk)
		unzip_path = os.path.join(tmpdir, "countries")
		unzip(download_path, unzip_path)

		shape_file_path = os.path.join(unzip_path, "WB_countries_Admin0_10m.shp")
		shape_file_data = geopandas.read_file(shape_file_path)
		shape_file_data.to_file(target_filename, driver='GeoJSON')


if __name__ == "__main__":
	try:
		target_filename = sys.argv[1]
	except IndexError:
		print(f"Usage: {sys.argv[0]} OUTPUT_GEOJSON_FILENAME", file=sys.stderr)
		sys.exit(1)

	if not target_filename.endswith('.geojson'):
		print("Expected target filename to end with .geojson", file=sys.stderr)
		sys.exit(1)

	try:
		download_country_polygons(target_filename)
	except DownloadError:
		print("Failed to download file", file=sys.stderr)
		sys.exit(1)
