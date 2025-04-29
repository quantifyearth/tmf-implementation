import argparse
import glob
import json
import os
import shutil
import tempfile
from typing import List
from http import HTTPStatus

import requests
import dotenv
from methods.common import DownloadError

dotenv.load_dotenv()
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

def download_granule(gedi_data_dir: str, name: str, url: str) -> None:
    os.makedirs(gedi_data_dir, exist_ok=True)
    final_name = os.path.join(gedi_data_dir, name)
    if os.path.exists(final_name):
        return
    print(f"Fetching {name}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with requests.Session() as session:
            if EARTHDATA_USER and EARTHDATA_PASSWORD:
                session.auth = (EARTHDATA_USER, EARTHDATA_PASSWORD)
            else:
                raise ValueError("Both EARTHDATA_USER and EARTHDATA_PASSWORD must be defined in environment.")
            auth_response = session.request('get', url)
            response = session.get(auth_response.url, auth=session.auth, stream=True)
            if response.status_code != HTTPStatus.OK:
                raise DownloadError(response.status_code, response.reason, url)
            download_target_name = os.path.join(tmpdir, name)
            with open(download_target_name, 'wb') as output_file:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    output_file.write(chunk)
        shutil.move(download_target_name, final_name)
    print(f"Downloading {name} complete.")

def gedi_fetch(granule_json_files: List[str], gedi_data_dir: str) -> None:
    for granule_json_file in granule_json_files:
        # skip totally empty files
        if os.path.getsize(granule_json_file) == 0:
            print(f"Skipping empty metadata file: {granule_json_file}")
            continue

        # try to load JSON
        try:
            with open(granule_json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON in {granule_json_file}: {e}")
            continue

        name = metadata.get("name")
        url  = metadata.get("url")
        if not name or not url:
            print(f"Skipping metadata missing name/url: {granule_json_file}")
            continue

        download_granule(gedi_data_dir, name, url)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GEDI granules from granule‐metadata JSON files"
    )
    parser.add_argument(
        "--granules",
        type=str,
        required=True,
        dest="granules_dir",
        help="Directory containing GEDI granule JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="gedi_data_dir",
        help="Directory to write downloaded GEDI data files"
    )
    args = parser.parse_args()

    # find all .json metadata files in granules dir
    files = sorted(glob.glob(os.path.join(args.granules_dir, "*.json")))
    if not files:
        print(f"No JSON files found in {args.granules_dir}")
        return

    try:
        gedi_fetch(files, args.gedi_data_dir)
    except DownloadError as exc:
        print(f"Failed to download: {exc.msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
