import argparse
import errno
import io
import logging
import math
import os
import shutil
import sys
import tempfile
import zipfile

import requests
import shapely # type: ignore
from geopandas import gpd # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

URL_TEMPLATE = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_%02d_%02d.zip"
TIFF_NAME_TEMPLATE = "srtm_%02d_%02d.tif"

def download_srtm_data(
    project_boundaries_filename: str,
    pixel_matching_boundaries_filename: str,
    destintation_tiff_folder: str
) -> None:
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

            tiff_target_name = TIFF_NAME_TEMPLATE % (xoffset, yoffset)
            tiff_target_path = os.path.join(destintation_tiff_folder, tiff_target_name)

            if not os.path.exists(tiff_target_path):
                with tempfile.TemporaryDirectory() as tempdir:
                    with requests.get(url, stream=True, timeout=60) as response:
                        if response.status_code == 404:
                            logging.warning("URL %s not found", url)
                            continue
                        response.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(response.content)) as zipf:
                            members = zipf.namelist()
                            for member in members:
                                unziped_path = zipf.extract(member, path=tempdir)
                                # The zip file has metadata files in it alongside the TIF we care about
                                if unziped_path.endswith(".tif"):
                                   shutil.move(unziped_path, tiff_target_path)

def main() -> None:
    # To not break the pipeline, I need to support the old non-named arguments and the new named version.
    # Unfortunately, AFAICT ArgumentParser's exit_on_error doesn't seem to be honoured if required arguments
    # aren't present, so we need to check sys.argv first before deciding how to handle args in the old or
    # new style.
    parser = argparse.ArgumentParser(description="Download SRTM data for a given set of areas.")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project",
        help="GeoTIFF file containing pixels in set M as generated by build_m_raster.py"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        dest="matching",
        help="GeoTIFF file containing pixels in set M as generated by build_m_raster.py"
    )
    parser.add_argument(
        "--tifs",
        type=str,
        required=True,
        dest="tif_folder_path",
        help="GeoTIFF file containing pixels in set M as generated by build_m_raster.py"
    )
    args = parser.parse_args()
    project_boundaries_filename = args.project
    pixel_matching_boundaries_filename = args.matching
    destination_tiff_folder = args.tif_folder_path

    download_srtm_data(
        project_boundaries_filename,
        pixel_matching_boundaries_filename,
        destination_tiff_folder
    )


if __name__ == "__main__":
    main()
