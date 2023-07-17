import os
import pathlib
import sys

from biomassrecovery.data.jrc_download import download_jrc_dataset  # type: ignore
from biomassrecovery.utils.unzip import unzip  # type: ignore


def download_jrc_data(target_zip_directory_str: str, target_tif_directory_str: str) -> None:

    target_zip_directory = pathlib.Path(target_zip_directory_str)
    target_tif_directory = pathlib.Path(target_tif_directory_str)

    download_jrc_dataset("AnnualChange", target_zip_directory)

    zips_dir = target_zip_directory / "AnnualChange"

    os.makedirs(target_tif_directory, exist_ok=True)

    for filename in os.listdir(zips_dir):
        if not filename.endswith('.zip'):
            continue

        zip_path = zips_dir / filename
        unzip(
            zip_path,
            target_tif_directory,
        )

def main() -> None:
    try:
        target_zip_directory = sys.argv[1]
        target_tif_directory = sys.argv[2]
    except IndexError:
        print(f"Usage: {sys.argv[0]} TARGET_ZIP_DOWNLOAD_DIRECTORY TARGET_TIF_DIRECTORY", file=sys.stderr)
        sys.exit(1)

    download_jrc_data(target_zip_directory, target_tif_directory)

if __name__ == "__main__":
    main()
