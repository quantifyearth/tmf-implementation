import glob
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np
from osgeo import gdal
from yirgacheffe.layers import RasterLayer
from yirgacheffe.window import Area, PixelScale

# Example filename: JRC_TMF_AnnualChange_v1_2011_AFR_ID37_N0_E40.tif
JRC_FILENAME_RE = re.compile(r".*_v1_(\d+)_.*_([NS]\d+)_([EW]\d+)\.tif")

def coursen_jrc_tile(tile_filename, result_filename, luc):
	# We rely on the fact that the JRC files are actually slightly overlapped and
	# just ignore the fact the boundaries won't quite line up for now. In theory though we should
	# be using multiple tiles to generate this
	src = RasterLayer.layer_from_file(tile_filename)
	filtered_result = RasterLayer.empty_raster_layer_like(src)
	def _filter(chunk):
		return np.asarray(chunk==luc)
	calc = src.numpy_apply(_filter)
	calc.save(filtered_result)

	result_pixel_scale = PixelScale(src.pixel_scale.xstep * 40, src.pixel_scale.ystep * 40)
	result_width = math.floor(src.window.xsize / 40.0)
	result_height = math.floor(src.window.ysize / 40.0)
	result_area = Area(
		top=src.area.top,
		left=src.area.left,
		bottom=src.area.top + (result_height * result_pixel_scale.ystep),
		right=src.area.left + (result_height * result_pixel_scale.xstep),
	)

	result_layer = RasterLayer.empty_raster_layer(
		result_area,
		result_pixel_scale,
		gdal.GDT_Float32,
		result_filename,
		src.projection
	)

	for y in range(result_height):
		buffer = []
		src = filtered_result.read_array(0, y * 40, result_width * 40, 40)
		for x in range(result_width):
			subset = src[0:40, x*40:(x+1)*40]
			buffer.append(subset.sum() / (40.0 * 40.0))
		result_layer._dataset.GetRasterBand(1).WriteArray(np.array([buffer]), 0, y)

def generate_coursened_luc(jrc_directory: str, result_directory: str) -> None:
	os.makedirs(result_directory, exist_ok=True)

	with tempfile.TemporaryDirectory() as tempdir:
		for filename in glob.glob("*.tif", root_dir=jrc_directory):
			match = JRC_FILENAME_RE.match(filename)
			if match is None:
				raise ValueError(f"Failed to parse JRC filename {filename}")
			year, x, y = match.groups()

			src = os.path.join(jrc_directory, filename)

			# Clearly more efficient to do this loop inside the coursen_jrc_tile
			# function, but we don't need to run this often and it keeps the code simpler
			for luc in range(1, 7):
				tempdest = os.path.join(tempdir, f"course_{x}_{y}_{year}_{luc}.tif")
				coursen_jrc_tile(src, tempdest, luc)

				shutil.move(tempdest, os.path.join(result_directory, f"course_{x}_{y}_{year}_{luc}.tif"))

if __name__ == "__main__":
	try:
		jrc_directory = sys.argv[1]
		result_directory = sys.argv[2]
	except IndexError:
		print(f"Usage: {sys.argv[0]} JRC_TILE_FOLDER COURSENED_LUC_FOLDER", file=sys.stderr)
		sys.exit(1)

	generate_coursened_luc(jrc_directory, result_directory)
