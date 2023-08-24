from glob import glob
import tempfile
import os
import subprocess
import argparse
import sys
import math

from methods.common.geometry import utm_for_geometry

import numpy as np
import geopandas as gpd
from yirgacheffe.layers import RasterLayer

def utm_code(lng):
    return  (math.floor((lng + 180.0) / 6.0) % 60) + 1

def generate_slope(
    input_elevation_directory: str,
    output_slope_directory: str
):
    elev = glob("*.tif", root_dir=input_elevation_directory)

    with tempfile.TemporaryDirectory() as tmpdir:
        for elevation_path in elev:
            reprojection = os.path.join(tmpdir, elevation_path)
            slope_path = os.path.join(tmpdir, "slope-" + elevation_path)
            elev_path = os.path.join(input_elevation_directory, elevation_path)
            os.makedirs(output_slope_directory, exist_ok=True)
            out_path = os.path.join(output_slope_directory, "slope-" + elevation_path)
            elevation = RasterLayer.layer_from_file(elev_path)


            lower_code, upper_code = range(utm_code(elevation.area.left), utm_code(elevation.area.right))

            # FAST PATH -- with only one UTM zone the reprojection back has no issues
            if lower_code == upper_code:
                utm_code = lower_code
                warp = f"gdalwarp -t_srs '+proj=utm +zone={utm_code} +datum=WGS84' {elev_path} {reprojection}"
                slope = f"gdaldem slope {reprojection} {slope_path}"
                warp_back = f"gdalwarp -t_srs '+proj=longlat +datum=WGS84' {slope_path} {out_path}"
                res = subprocess.call(warp, shell=True)
                if res != 0:
                    exit(res)
                res = subprocess.call(slope, shell=True)
                if res != 0:
                    exit(res)
                res = subprocess.call(warp_back, shell=True)
                if res != 0:
                    exit(res)
            # SLOW PATH -- in the slow path, we have to break the elevation raster into
            # UTM sections and do the above to each before reprojecting back and recombining
            for utm_code in range(lower_code, upper_code + 1):
                area = 

def main() -> None:
    parser = argparse.ArgumentParser(description="Generates expanded boundary shape for a project")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        dest="input_elevations",
        help="Directory of input elevation tifs."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_slopes",
        help="Output directory for the slopes."
    )
    args = parser.parse_args()

    try:
        generate_slope(args.input_elevations, args.output_slopes)
    except FileNotFoundError as exc:
        print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()