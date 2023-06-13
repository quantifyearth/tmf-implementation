import json

import pytest
from osgeo import ogr, osr

from methods.inputs.generate_boundary import utm_for_geometry, expand_boundaries # pylint: disable=E0401

@pytest.mark.parametrize(
    "lat,lng,expected",
    [
        (52.205276, 0.119167, 32631), # Cambridge
        (-52.205276, 0.119167, 32731), # Anti-Cambridge
        (6.152938, 38.202316, 32637), # Yirgacheffe
        (4.710989, -74.072090, 32618), # Bogotá
    ]
)
def test_utm_band(lat, lng, expected):
    origin_lat = lat + 0.2
    origin_lng = lng - 0.2
    far_lat = lat - 0.2
    far_lng = lng + 0.2

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
    test_poly = ogr.CreateGeometryFromJson(json.dumps(frame))

    utm_code = utm_for_geometry(test_poly)
    assert utm_code == expected

@pytest.mark.parametrize(
    "lat,lng",
    [
        (52.205276, 0.119167), # Cambridge
        (-52.205276, 0.119167), # Anti-Cambridge
        (6.152938, 38.202316), # Yirgacheffe
        (4.710989, -74.072090), # Bogotá
    ]
)
def test_expand_band(lat, lng):
    origin_lat = lat + 0.2
    origin_lng = lng - 0.2
    far_lat = lat - 0.2
    far_lng = lng + 0.2

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
    test_poly = ogr.CreateGeometryFromJson(json.dumps(frame))

    original_area = test_poly.GetArea()

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326) # aka WSG84
    test_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
    test_layer = test_data_source.CreateLayer("buffer", spatial_ref, geom_type=ogr.wkbMultiPolygon)
    feature_definition = test_layer.GetLayerDefn()
    new_feature = ogr.Feature(feature_definition)
    new_feature.SetGeometry(test_poly)
    test_layer.CreateFeature(new_feature)

    expanded_dataset = expand_boundaries(test_layer, 1000)
    assert len(expanded_dataset) == 1
    expanded_layer = expanded_dataset.GetLayer()
    assert len(expanded_layer) == 1
    expanded_feature = expanded_layer.GetNextFeature()
    expanded_geometry = expanded_feature.GetGeometryRef()
    expanded_area = expanded_geometry.GetArea()

    assert expanded_area > original_area
