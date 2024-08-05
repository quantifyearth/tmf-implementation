import argparse
import io
import logging
import multiprocessing
import os
import shutil
import tempfile
import zipfile
from functools import partial
from http import HTTPStatus
from typing import Optional

import geopandas as gpd # type: ignore
import requests
import shapely # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

URL_FORMAT = "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange&lat=%s&lon=%s"

def download_jrc_dataset(tempdir: str, output_dir: str, tile_url: str) -> None:
    with requests.get(tile_url, stream=True, timeout=60) as response:
        if response.status_code == HTTPStatus.NOT_FOUND:
            logging.warning("No file found at %s", tile_url)
            return
        if response.status_code != HTTPStatus.OK:
            logging.warning("Bad status for %s: %d", tile_url, response.status_code)
            raise ValueError(response.status_code)
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zzip:
                members = zzip.namelist()
                for member in members:
                    target = zzip.extract(member, path=tempdir)
                    _, basename = os.path.split(target)
                    final_path = os.path.join(output_dir, basename)
                    try:
                        os.rename(target, final_path)
                    except OSError:
                        shutil.move(target, final_path)
            logging.info("Successfully fetched %s", tile_url)
        except zipfile.BadZipFile:
            logging.error("Bad zip file %s: %s", tile_url, response.content)
            raise

def download_jrc_data(
    target_tif_directory: str,
    boundary_filename: Optional[str],
) -> None:

    os.makedirs(target_tif_directory, exist_ok=True)

    boundary = gpd.read_file(boundary_filename) if boundary_filename else None

    tile_urls = []
    for lat in range(30, -30, -10):
        for lng in range(-110, 180, 10):
            if boundary is not None:
                tilearea = shapely.Polygon((
                    (lng, lat),
                    (lng, lat - 10),
                    (lng + 10, lat - 10),
                    (lng + 10, lat),
                ))
                i = shapely.intersects(tilearea, boundary.geometry.values[0])
                if not i:
                    continue

            slat = f"S{lat * -1}" if lat < 0 else f"N{lat}"
            slng = f"W{lng * -1}" if lng < 0 else f"E{lng}"

            url = URL_FORMAT % (slat, slng)
            tile_urls.append(url)

    logging.info("Found %s tiles", len(tile_urls))
    with tempfile.TemporaryDirectory() as tempdir:
        core_count = int(multiprocessing.cpu_count() / 4)
        n_workers = min(len(tile_urls), core_count)
        with multiprocessing.Pool(n_workers) as pool:
            pool.map(partial(download_jrc_dataset, tempdir, target_tif_directory), tile_urls)

def main() -> None:
    parser = argparse.ArgumentParser(description="Download JRC tiles, optional constrained to an area.")
    parser.add_argument(
        "--boundary",
        type=str,
        required=False,
        dest="boundary_filename",
        help="GeoJSON of area boundary."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="target_tif_directory",
        help="Where to store the final tiffs."
    )
    args = parser.parse_args()

    download_jrc_data(
        args.target_tif_directory,
        args.boundary_filename
    )

if __name__ == "__main__":
    main()
