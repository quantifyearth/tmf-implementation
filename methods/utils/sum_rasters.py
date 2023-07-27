import glob
import os
import sys

from osgeo import gdal  # type: ignore
from yirgacheffe.layers import RasterLayer  # type: ignore

import yirgacheffe.operators # type: ignore
yirgacheffe.operators.YSTEP = 1024 * 8

def sum_rasters(input_folder: str, output_raster: str) -> None:
	rasters = [os.path.join(input_folder, filename) for filename in glob.glob("*.tif", root_dir=input_folder)]

	temp_raster = RasterLayer.layer_from_file(rasters[0])
	for raster_file in rasters[1:]:
		this_raster = RasterLayer.layer_from_file(raster_file)
		new_res = RasterLayer.empty_raster_layer_like(temp_raster, datatype=gdal.GDT_UInt16)
		calc = this_raster + temp_raster
		calc.save(new_res)
		temp_raster = new_res

	final = RasterLayer.empty_raster_layer_like(temp_raster, output_raster)
	temp_raster.save(final)

if __name__ == "__main__":
	try:
		input_folder = sys.argv[1]
		output_raster = sys.argv[2]
	except IndexError:
		print(f"Usage: {sys.argv[0]} INPUT_FOLDER OUTPUT_RASTER", file=sys.stderr)
		sys.exit(1)

	sum_rasters(input_folder, output_raster)
