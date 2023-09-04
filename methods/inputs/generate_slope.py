from glob import glob
import tempfile
import os
import subprocess
import argparse
import sys
import math
import logging

import utm # type: ignore
from yirgacheffe.window import Area # type: ignore
from yirgacheffe.layers import RasterLayer, GroupLayer # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def utm_code(lng):
    return (math.floor((lng + 180.0) / 6.0) % 60) + 1


# Truncated to not include the wonky European parts of UTM
UTM_LETTERS = [
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
]


def crange(utm_code_1, utm_code_2):
    start = UTM_LETTERS.index(utm_code_1)
    last = UTM_LETTERS.index(utm_code_2)
    for utm_code_idx in range(start, last + 1):
        yield UTM_LETTERS[utm_code_idx]

# Note this only correct for the majority of UTM zones, but hopefully
# we never have to deal with Norway...
def bounding_box_of_utm(zone: int, letter: str):
    upper_lng = zone * 6 - 180
    lower_lng = upper_lng - 6
    if letter not in UTM_LETTERS:
        raise ValueError(f"Slope calculation is not support in UTM latitude {letter}")
    lower_lat = UTM_LETTERS.index(letter) * 8 - 80
    upper_lat = lower_lat + 8
    return Area(left=lower_lng, right=upper_lng, bottom=lower_lat, top=upper_lat)


def warp(
    utm_zone: int,
    elev_path: str,
    reprojection_path: str,
    slope_path: str,
    out_path: str,
):
    warp_cmd = f"gdalwarp -t_srs '+proj=utm +zone={utm_zone} +datum=WGS84' {elev_path} {reprojection_path}"
    slope = f"gdaldem slope {reprojection_path} {slope_path}"
    warp_back = f"gdalwarp -t_srs '+proj=longlat +datum=WGS84' {slope_path} {out_path}"
    res = subprocess.call(warp_cmd, shell=True)
    if res != 0:
        logging.warning("Failed to run %s", warp_cmd)
        sys.exit(res)
    res = subprocess.call(slope, shell=True, close_fds=True)
    if res != 0:
        logging.warning("Failed to run %s", slope)
        sys.exit(res)
    res = subprocess.call(warp_back, shell=True, close_fds=True)
    if res != 0:
        logging.warning("Failed to run %s", warp_back)
        sys.exit(res)

def generate_slope(input_elevation_directory: str, output_slope_directory: str):
    elev = glob("*.tif", root_dir=input_elevation_directory)

    for elevation_path in elev:
        with tempfile.TemporaryDirectory() as tmpdir:
            elev_path = os.path.join(input_elevation_directory, elevation_path)
            os.makedirs(output_slope_directory, exist_ok=True)
            out_path = os.path.join(output_slope_directory, "slope-" + elevation_path)
            elevation = RasterLayer.layer_from_file(elev_path)

            logging.info("Area of elevation tile %a", elevation.area)
            _easting, _northing, lower_code, lower_letter = utm.from_latlon(
                elevation.area.bottom, elevation.area.left
            )
            _easting, _northing, upper_code, upper_letter = utm.from_latlon(
                elevation.area.top, elevation.area.right
            )

            # FAST PATH -- with only one UTM zone the reprojection back has no issues
            if lower_code == upper_code and lower_letter == upper_letter:
                actual_utm_code = lower_code
                reprojection_path = os.path.join(tmpdir, elevation_path)
                slope_path = os.path.join(tmpdir, "slope-" + elevation_path)
                warp(actual_utm_code, elev_path, reprojection_path, slope_path, out_path)
            else:
                # SLOW PATH -- in the slow path, we have to break the elevation raster into
                # UTM sections and do the above to each before reprojecting back and recombining

                # To capture the results here for later inspection just override the tmpdir variable
                for actual_utm_code in range(lower_code, upper_code + 1):
                    for utm_letter in crange(lower_letter, upper_letter):
                        logging.info("UTM(%s,%s)", actual_utm_code, utm_letter)
                        bbox = bounding_box_of_utm(actual_utm_code, utm_letter)
                        utm_layer = RasterLayer.empty_raster_layer_like(
                            elevation, area=bbox
                        )
                        utm_id = f"{actual_utm_code}-{utm_letter}-{elevation_path}"
                        utm_clip_path = os.path.join(tmpdir, utm_id)
                        intersection = RasterLayer.find_intersection(
                            [elevation, utm_layer]
                        )
                        result = RasterLayer.empty_raster_layer(
                            intersection,
                            elevation.pixel_scale,
                            elevation.datatype,
                            utm_clip_path,
                            elevation.projection,
                        )
                        result.set_window_for_intersection(intersection)
                        elevation.set_window_for_intersection(intersection)
                        elevation.save(result)

                        # Flush elevation utm clip to disk
                        del result

                        reprojection_path = os.path.join(tmpdir, "reproject-" + utm_id)
                        slope_path = os.path.join(tmpdir, "slope-" + utm_id)
                        slope_out_path = os.path.join(tmpdir, "out-slope-" + utm_id)
                        warp(
                            actual_utm_code,
                            utm_clip_path,
                            reprojection_path,
                            slope_path,
                            slope_out_path,
                        )
                # Now to recombine the UTM gridded slopes into the slope tile
                slopes = glob("out-slope-*", root_dir=tmpdir)
                assert len(slopes) > 0

                # This sets the order a little better for the union of the layers
                slopes.sort()
                slopes.reverse()

                logging.info("Render order %s", slopes)

                combined = GroupLayer(
                    [
                        RasterLayer.layer_from_file(os.path.join(tmpdir, filename))
                        for filename in slopes
                    ]
                )

                elevation = RasterLayer.layer_from_file(elev_path)
                intersection = RasterLayer.find_intersection([elevation, combined])
                result = RasterLayer.empty_raster_layer(
                    intersection,
                    elevation.pixel_scale,
                    elevation.datatype,
                    out_path,
                    elevation.projection,
                )
                combined.set_window_for_intersection(intersection)
                result.set_window_for_intersection(intersection)
                combined.save(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generates expanded boundary shape for a project"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        dest="input_elevations",
        help="Directory of input elevation tifs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_slopes",
        help="Output directory for the slopes.",
    )
    args = parser.parse_args()

    try:
        generate_slope(args.input_elevations, args.output_slopes)
    except FileNotFoundError as exc:
        print(f"Failed to find file {exc.filename}: {exc.strerror}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
