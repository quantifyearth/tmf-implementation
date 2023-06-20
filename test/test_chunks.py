import pytest
from osgeo import ogr, osr

from methods.inputs.download_gedi_data import chunk_geometry # pylint: disable=E0401
from .helpers import build_polygon


@pytest.mark.parametrize(
	"diameter,chunk_size,expected_count",
	[
		(0.1, 0.2, 1),
		(0.2, 0.2, 1),
		(0.2, 0.1, 4),
		(0.4, 0.1, 16),
	]
)
def test_chunk_large_area(diameter: float, chunk_size: float, expected_count: int) -> None:
	test_poly = build_polygon(42.3, 12.6, diameter/2)

	spatial_ref = osr.SpatialReference()
	spatial_ref.ImportFromEPSG(4326) # aka WSG84
	test_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
	test_layer = test_data_source.CreateLayer("buffer", spatial_ref, geom_type=ogr.wkbMultiPolygon)
	feature_definition = test_layer.GetLayerDefn()
	new_feature = ogr.Feature(feature_definition)
	new_feature.SetGeometry(test_poly)
	test_layer.CreateFeature(new_feature)

	chunked_datasource = chunk_geometry(test_layer, chunk_size)
	chunked_layer = chunked_datasource.GetLayer()
	assert len(chunked_layer) == expected_count
