import errno
import math
import os
import shutil
import sys
import tempfile
import logging

import requests
import shapely # type: ignore
from fiona.errors import DriverError # type: ignore
from geopandas import gpd # type: ignore

from biomassrecovery.utils.unzip import unzip # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

URL_TEMPLATE = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip"
TIFF_NAME_TEMPLATE = "srtm_%02d_%02d.tif"

def download_srtm_data(
    project_boundaries_filename: str,
    pixel_matching_boundaries_filename: str,
    destination_zip_folder: str,
    destintation_tiff_folder: str
) -> None:
    os.makedirs(destination_zip_folder, exist_ok=True)
    os.makedirs(destintation_tiff_folder, exist_ok=True)

    project = gpd.read_file(project_boundaries_filename)
    matching_area = gpd.read_file(pixel_matching_boundaries_filename)
    total = shapely.union(project.geometry, matching_area.geometry)
    min_x, min_y, max_x, max_y = total.envelope[0].bounds

    min_x = math.floor((180.0 + min_x) / 5.0) + 1
    max_x = math.floor((180.0 + max_x) / 5.0) + 1
    min_y = math.floor((60.0 - min_y) / 5.0) + 1
    max_y = math.floor((60.0 - max_y) / 5.0) + 1

    if min_y > max_y:
        min_y, max_y = max_y, min_y

    for yoffset in range(min_y, max_y + 1):
        for xoffset in range(min_x, max_x + 1):
            url = URL_TEMPLATE % (xoffset, yoffset)
            logging.info("Fetching SRTM tile %s", url)
            target_filename = url.split('/')[-1]
            target_path = os.path.join(destination_zip_folder, target_filename)

            if not os.path.exists(target_path):
                with tempfile.TemporaryDirectory() as tempdir:
                    download_target = os.path.join(tempdir, target_filename)
                    with requests.get(url, stream=True, timeout=60) as response:
                        # We are pretty coarse with our max x and y and so we might
                        # create a big rectangle that includes a tile that is all ocean
                        # and it is easier to just drop 404s.
                        if response.status_code == 404:
                            logging.warning("URL %s not found", url)
                            continue
                        response.raise_for_status()
                        with open(download_target, 'wb') as output_file:
                            for chunk in response.iter_content(chunk_size=1024*1024):
                                output_file.write(chunk)
                        shutil.move(download_target, target_path)

            tiff_target_name = TIFF_NAME_TEMPLATE % (xoffset, yoffset)
            tiff_target_path = os.path.join(destintation_tiff_folder, tiff_target_name)
            if not os.path.exists(tiff_target_path):
                with tempfile.TemporaryDirectory() as tempdir:
                    unzip(target_path, tempdir)
                    unziped_tiff_path = os.path.join(tempdir, tiff_target_name)
                    if not os.path.exists(unziped_tiff_path):
                        raise FileNotFoundError(errno.ENOENT, "Zip contents not as expected", tiff_target_name)
                    shutil.move(unziped_tiff_path, tiff_target_path)

def main() -> None:
    try:
        project_boundaries_filename = sys.argv[1]
        pixel_matching_boundaries_filename = sys.argv[2]
        destination_zip_folder = sys.argv[3]
        destination_tiff_folder = sys.argv[4]
    except IndexError:
        print(f"Usage: {sys.argv[0]} PROJECT_BOUNDARIES PIXEL_MATCHING_AREA "
            "DESTINATION_ZIP_FOLDER DESTINATION_TIFF_FOLDER", file=sys.stderr)
        sys.exit(1)

    try:
        download_srtm_data(
            project_boundaries_filename,
            pixel_matching_boundaries_filename,
            destination_zip_folder,
            destination_tiff_folder
        )
    except DriverError as exc:
        print(exc.args[0], file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
