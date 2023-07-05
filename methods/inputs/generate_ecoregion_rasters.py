import glob
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
from functools import partial

from yirgacheffe.layers import RasterLayer, VectorLayer

# The raw geojson for ecoregions is over 600MG, and OGR (which even geopandas uses under the hood)
# will throw an error when it hits 200MB unless you override the limit thus
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

def process_tile(result_path, jrc_path) -> None:
	with tempfile.TemporaryDirectory() as tempdir:
		jrc_raster = RasterLayer.layers_from_file(jrc_path)
		ecoregions = VectorLayer.layer_from_file(ecoregions_filename, None, jrc_raster.pixel_scale, jrc_raster.projection, burn_value="ECO_ID")

		_, lat, lng = re.match(".*_([NS]\d+)_([WE]\d+).tif", jrc_path)

		filename = f"ecoregion_{lat}_{lng}.tif"
		target_filename = os.path.join(tempdir, filename)
		result = Raster.empty_raster_layer_file(jrc_raster, filename=target_filename)
		ecoregions.set_window_for_intersection(jrc_raster.area)
		ecoregions.save(result)
		shutil.move(target, os.path.join(result_path, filename))

def generate_ecoregion_rasters(
	ecoregions_filename: str,
	jrc_folder: str,
	output_folder: str,
) -> None:
	os.makedirs(output_folder, exists_ok=True)
	jrc_files = [os.path.join(jrc_folder, filename) for filename in glob.glob("*2020*.tif", root_dir=jrc_folder)]
	with Pool(processes=len(jrc_files)) as p:
		p.map(jrc_files, partial(process_tile, output_folder))

if __name__ == "__main__":
	try:
		ecoregions_filename = sys.argv[1]
		jrc_folder = sys.argv[2]
		output_folder = sys.argv[3]
	except IndexError:
		print(f"Usage: {sys.argv[0]} ECOREGIONS_GEOJSON JRC_FOLDER OUTPUT_FOLDER", file=sys.stderr)
		sys.exit(1)

	generate_ecoregion_rasters(ecoregions_filename, jrc_folder, output_folder)