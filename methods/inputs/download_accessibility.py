import os
import sys
import tempfile
from glob import glob
import requests

from biomassrecovery.utils.unzip import unzip  # type: ignore

# As taken from https://malariaatlas.org/open-access-policy/
# Under "Creative Commons Attribution 3.0 Unported License"
ACCESS_DATA = "https://data.malariaatlas.org/geoserver/Accessibility/ows?service=CSW&version=2.0.1&request=DirectDownload&ResourceId=Accessibility:202001_Global_Walking_Only_Travel_Time_To_Healthcare"

class DownloadError(Exception):
	"""Indicate the download failed"""

def is_tif(fname : str) -> bool:
	return os.path.splitext(fname)[1] in ['.tif', '.tiff']

def download_accessibility_tif(source_url: str, target_path: str) -> None:
	with tempfile.TemporaryDirectory() as tmpdir:
		download_path = os.path.join(tmpdir, "accessibility.zip")
		response = requests.get(source_url, stream=True)
		if not response.ok:
			raise DownloadError(f'{response.status_code}: {response.reason}')
		with open(download_path, 'wb') as fd:
			print(f"Downloading accessibility data...")
			for chunk in response.iter_content(chunk_size=1024*1024):
				fd.write(chunk)
		print(f"Unzipping from {download_path}")
		unzip(
			download_path,
			tmpdir,
			filter_func=is_tif
		)

		# There should only be a single TIF file
		tif = glob("*.tif", root_dir=tmpdir)
		tiff = glob("*.tiff", root_dir=tmpdir)

		tifs = tif + tiff

		if len(tifs) != 1:
			tif_names = " ".join(tifs)
			raise ValueError(f"Downloading accessiblity data should only result in a single TIF\nGot: {tif_names}")

		os.rename(os.path.join(tmpdir, tifs[0]), target_path)


if __name__ == "__main__":
	try:
		target_filename = sys.argv[1]
	except IndexError:
		print(f"Usage: {sys.argv[0]} OUTPUT_TIF_FILENAME", file=sys.stderr)
		sys.exit(1)

	try:
		download_accessibility_tif(ACCESS_DATA, target_filename)
	except DownloadError:
		print("Failed to download file", file=sys.stderr)
		sys.exit(1)