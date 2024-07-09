import json
import os
import shutil
import sys
import tempfile
from http import HTTPStatus

import requests
import dotenv
from methods.common import DownloadError

# This is defined in biomassrecovery.environment too, but that file
# is full of side-effects, so just import directly here.
dotenv.load_dotenv()
EARTHDATA_USER = os.getenv("EARTHDATA_USER")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

def download_granule(gedi_data_dir: str, name: str, url: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with requests.Session() as session:
            if EARTHDATA_USER and EARTHDATA_PASSWORD:
                session.auth = (EARTHDATA_USER, EARTHDATA_PASSWORD)
            else:
                raise ValueError("Both EARTHDATA_USER and EARTHDATA_PASSWORD must be defined in environment.")
            # Based on final example in https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
            # you have to make an initial request and then auth on that
            auth_response = session.request('get', url)
            response = session.get(auth_response.url, auth=session.auth, stream=True)
            if response.status_code != HTTPStatus.OK:
                raise DownloadError(response.status_code, response.reason, url)
            download_target_name = os.path.join(tmpdir, name)
            with open(download_target_name, 'wb') as output_file:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    output_file.write(chunk)

        final_name = os.path.join(gedi_data_dir, name)
        shutil.move(download_target_name, final_name)

def gedi_fetch(granule_json_file: str, gedi_data_dir: str) -> None:
    with open(granule_json_file, "r", encoding="utf-8") as f:
        metadata = json.loads(f.read())
    download_granule(gedi_data_dir, metadata["name"], metadata["url"])

def main() -> None:
    try:
        granule_json_file = sys.argv[1]
        gedi_data_dir = sys.argv[2]
    except IndexError:
        print(f"Usage: {sys.argv[0]} BUFFER_BOUNDRY_FILE GEDI_DATA_DIRECTORY")
        sys.exit(1)
    except DownloadError as exc:
        print(f"Failed to download: {exc.msg}")
        sys.exit(1)

    gedi_fetch(granule_json_file, gedi_data_dir)

if __name__ == "__main__":
    main()
