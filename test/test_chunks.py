import json
import pytest
from osgeo import ogr, osr # type: ignore

from methods.inputs.locate_gedi_data import chunk_geometry # pylint: disable=E0401

def _build_polygon(lat: float, lng: float, radius: float) -> ogr.Geometry:
    origin_lat = lat + radius
    origin_lng = lng - radius
    far_lat = lat - radius
    far_lng = lng + radius

    frame = {
        'type': 'POLYGON',
        'coordinates': [
            [
                [origin_lng, origin_lat],
                [far_lng,    origin_lat],
                [far_lng,    far_lat],
                [origin_lng, far_lat],
                [origin_lng, origin_lat],
            ]
        ]
    }
    return ogr.CreateGeometryFromJson(json.dumps(frame))

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
    test_poly = _build_polygon(42.3, 12.6, diameter/2)

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
