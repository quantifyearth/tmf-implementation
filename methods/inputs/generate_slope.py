import argparse
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob

import utm  # type: ignore
from yirgacheffe.window import Area  # type: ignore
from yirgacheffe.layers import RasterLayer, GroupLayer  # type: ignore

UTM_EXPANSION_DEGREES = 0.3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def utm_code(lng: float) -> float:
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
def bounding_box_of_utm(zone: int, letter: str, expansion: float) -> Area:
    upper_lng = zone * 6 - 180
    lower_lng = upper_lng - 6
    if letter not in UTM_LETTERS:
        raise ValueError(f"Slope calculation is not support in UTM latitude {letter}")
    lower_lat = UTM_LETTERS.index(letter) * 8 - 80
    upper_lat = lower_lat + 8
    return Area(
        left=lower_lng - expansion,
        right=upper_lng + expansion,
        bottom=lower_lat - expansion,
        top=upper_lat + expansion,
    )


def warp(
    utm_zone: int,
    elev_path: str,
    pixel_scale_x: float,
    pixel_scale_y: float,
    out_path: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        reprojection_path = os.path.join(tmpdir, "reprojection.tif")
        slope_path = os.path.join(tmpdir, "sloped.tif")
        final_reprojection_path = os.path.join(tmpdir, "final.tif")

        warp_cmd = f"gdalwarp -t_srs '+proj=utm +zone={utm_zone} +datum=WGS84' {elev_path} {reprojection_path}"
        slope = f"gdaldem slope {reprojection_path} {slope_path}"
        warp_back = f"gdalwarp -tr {pixel_scale_x} {pixel_scale_y} -t_srs \
                    '+proj=longlat +datum=WGS84' {slope_path} {final_reprojection_path}"

        res = subprocess.call(warp_cmd, shell=True)
        if res != 0:
            raise ValueError(f"Failed to run {warp_cmd} exited {res}")
        res = subprocess.call(slope, shell=True, close_fds=True)
        if res != 0:
            raise ValueError(f"Failed to run {slope} exited {res}")
        res = subprocess.call(warp_back, shell=True, close_fds=True)
        if res != 0:
            raise ValueError(f"Failed to run {warp_back} exited {res}")

        shutil.move(final_reprojection_path, out_path)


def generate_slope(input_elevation_directory: str, output_slope_directory: str):
    elev = glob("*.tif", root_dir=input_elevation_directory)
    os.makedirs(output_slope_directory, exist_ok=True)

    for elevation_path in elev:
        elev_path = os.path.join(input_elevation_directory, elevation_path)
        out_path = os.path.join(output_slope_directory, "slope-" + elevation_path)
        if os.path.exists(out_path):
            logging.info("%s already exists, skipping.", out_path)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
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
                warp(
                    actual_utm_code,
                    elev_path,
                    elevation.pixel_scale.xstep,
                    elevation.pixel_scale.ystep,
                    out_path,
                )
            else:
                # SLOW PATH -- in the slow path, we have to break the elevation raster into
                # UTM sections and do the above to each before reprojecting back and recombining

                # To capture the results here for later inspection just override the tmpdir variable
                for actual_utm_code in range(lower_code, upper_code + 1):
                    for utm_letter in crange(lower_letter, upper_letter):
                        logging.debug("UTM(%s,%s)", actual_utm_code, utm_letter)

                        # Note: we go a little bit around the UTM tiles and will crop them down to size later
                        # this is to remove some aliasing effects.
                        bbox = bounding_box_of_utm(actual_utm_code, utm_letter, UTM_EXPANSION_DEGREES)

                        # Crop the elevation tile to a UTM zone
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

                        # Now warp into UTM, calculate slopes, and warp back
                        slope_out_path = os.path.join(tmpdir, "out-slope-" + utm_id)
                        warp(
                            actual_utm_code,
                            utm_clip_path,
                            elevation.pixel_scale.xstep,
                            elevation.pixel_scale.ystep,
                            slope_out_path,
                        )

                        # We now recrop the out-slope back to the bounding box we assumed at the start
                        bbox_no_expand = bounding_box_of_utm(
                            actual_utm_code, utm_letter, 0.0
                        )
                        slope_tif = RasterLayer.layer_from_file(slope_out_path)
                        grid = RasterLayer.empty_raster_layer_like(
                            slope_tif, area=bbox_no_expand
                        )
                        output_final = f"final-slope-{actual_utm_code}-{utm_letter}-{elevation_path}"
                        final_path = os.path.join(tmpdir, output_final)
                        logging.debug("Slope underlying %s", slope_tif._underlying_area) # pylint: disable=W0212
                        logging.debug("Grid underling %s", grid._underlying_area) # pylint: disable=W0212
                        try:
                            intersection = RasterLayer.find_intersection([slope_tif, grid])
                        except ValueError:
                            logging.debug(
                                "UTM (%s, %s) didn't intersect actual area %s",
                                actual_utm_code,
                                utm_letter,
                                grid._underlying_area # pylint: disable=W0212
                            )
                            continue
                        slope_tif.set_window_for_intersection(intersection)
                        final = RasterLayer.empty_raster_layer(
                            intersection,
                            slope_tif.pixel_scale,
                            slope_tif.datatype,
                            final_path,
                            slope_tif.projection,
                        )
                        logging.debug("Final underlying %s", final._underlying_area) # pylint: disable=W0212
                        final.set_window_for_intersection(intersection)
                        slope_tif.save(final)

                        # Flush
                        del final

                # Now to recombine the UTM gridded slopes into the slope tile
                slopes = glob("final-slope-*", root_dir=tmpdir)
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
                combined.set_window_for_intersection(intersection)
                elevation.set_window_for_intersection(intersection)

                assembled_path = os.path.join(tmpdir, "patched.tif")
                result = RasterLayer.empty_raster_layer_like(
                    elevation, filename=assembled_path
                )
                combined.save(result)

                shutil.move(assembled_path, out_path)


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
