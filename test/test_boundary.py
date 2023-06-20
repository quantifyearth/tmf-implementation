import json
from typing import List

import pytest
from osgeo import ogr, osr # type: ignore

from methods.inputs.generate_boundary import utm_for_geometry, expand_boundaries # pylint: disable=E0401
from .helpers import build_polygon

@pytest.mark.parametrize(
    "lat,lng,expected",
    [
        (52.205276, 0.119167, 32631), # Cambridge
        (-52.205276, 0.119167, 32731), # Anti-Cambridge
        (6.152938, 38.202316, 32637), # Yirgacheffe
        (4.710989, -74.072090, 32618), # Bogotá
    ]
)
def test_utm_band(lat: float, lng: float, expected: int) -> None:
    test_poly = build_polygon(lat, lng, 0.2)

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
def test_expand_boundary(lat: float, lng: float) -> None:
    test_poly = build_polygon(lat, lng, 0.2)

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


@pytest.mark.parametrize(
    "polygon_list,expected_count",
    [
        # Just one polygon
        (
            {(10.0, 10.0, 0.2)}, 1,
        ),
        # Pure superset
        (
            {(10.0, 10.0, 0.2), (10.0, 10.0, 0.1)}, 1,
        ),
        # Distinct
        (
            {(10.0, 10.0, 0.2), (11.0, 11.0, 0.2)}, 2,
        ),
        # Overlap
        (
            {(10.0, 10.0, 0.2), (10.3, 10.3, 0.2)}, 1,
        ),
    ]
)
def test_simplify_output_geometry(polygon_list, expected_count):
    def _make_square(lat: float, lng: float, radius: float) -> List[List[List[float]]]:
        origin_lat = lat + radius
        origin_lng = lng - radius
        far_lat = lat - radius
        far_lng = lng + radius
        return [[
            [origin_lng, origin_lat],
            [far_lng,    origin_lat],
            [far_lng,    far_lat],
            [origin_lng, far_lat],
            [origin_lng, origin_lat],
        ]]

    frame = {
        'type': 'MULTIPOLYGON',
        'coordinates': [_make_square(*poly) for poly in polygon_list]
    }
    test_multipoly = ogr.CreateGeometryFromJson(json.dumps(frame))
    assert test_multipoly.GetGeometryType() == ogr.wkbMultiPolygon
    assert test_multipoly.GetGeometryCount() == len(polygon_list)

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326) # aka WSG84
    test_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
    test_layer = test_data_source.CreateLayer("buffer", spatial_ref, geom_type=ogr.wkbMultiPolygon)
    feature_definition = test_layer.GetLayerDefn()
    new_feature = ogr.Feature(feature_definition)
    new_feature.SetGeometry(test_multipoly)
    test_layer.CreateFeature(new_feature)

    expanded_dataset = expand_boundaries(test_layer, 1000)
    assert len(expanded_dataset) == 1

    expanded_layer = expanded_dataset.GetLayer()
    assert len(expanded_layer) == 1

    expanded_feature = expanded_layer.GetNextFeature()
    assert expanded_layer.GetNextFeature() is None

    expanded_geometries = expanded_feature.GetGeometryRef()

    assert expanded_geometries.GetGeometryType() == ogr.wkbMultiPolygon if expected_count > 1 else ogr.wkbPolygon
    assert expanded_geometries.GetGeometryCount() == expected_count
