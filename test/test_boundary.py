import json
from typing import List, Set, Tuple

import pytest
from osgeo import ogr, osr # type: ignore

from methods.common.geometry import utm_for_geometry, find_overlapping_geometries, expand_boundaries
from .helpers import build_polygon, build_datasource, build_multipolygon

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
def test_simplify_output_geometry(polygon_list: Set[Tuple[float]], expected_count: int) -> None:
    test_multipoly = build_multipolygon(polygon_list)
    test_data_source = build_datasource([test_multipoly])
    test_layer = test_data_source.GetLayer()

    expanded_dataset = expand_boundaries(test_layer, 1000)
    assert len(expanded_dataset) == 1

    expanded_layer = expanded_dataset.GetLayer()
    assert len(expanded_layer) == 1

    expanded_feature = expanded_layer.GetNextFeature()
    assert expanded_layer.GetNextFeature() is None

    expanded_geometries = expanded_feature.GetGeometryRef()

    assert expanded_geometries.GetGeometryType() == ogr.wkbMultiPolygon if expected_count > 1 else ogr.wkbPolygon
    assert expanded_geometries.GetGeometryCount() == expected_count

@pytest.mark.parametrize(
    "src_list,guard_list,expected",
    [
        (
            {(10.0, 10.0, 0.2)},
            {(10.0, 20.0, 0.2)},
            0,
        ),
        (
            {(10.0, 10.0, 2.0)},
            {(11.0, 10.0, 2.0)},
            1,
        ),
        (
            {(10.0, 10.0, 2.0), (20, 20, 1.0)},
            {(11.0, 10.0, 2.0)},
            1,
        ),
        (
            {(10.0, 10.0, 2.0)},
            {(11.0, 10.0, 2.0), (9.0, 10.0, 2.0)},
            1,
        ),
        (
            {(10.0, 10.0, 2.0), (20, 20, 1.0)},
            {(11.0, 10.0, 10.0)},
            2,
        ),
    ]
)
def test_overlapping_geometries(src_list: Set[Tuple[float]], guard_list: Set[Tuple[float]], expected: int) -> None:
    srcs = [build_polygon(*x) for x in src_list]
    src = build_datasource(srcs)
    guards = [build_polygon(*x) for x in guard_list]
    guard = build_datasource(guards)

    result = find_overlapping_geometries(src.GetLayer(), guard.GetLayer())

    result_layer = result.GetLayer()
    assert len(result_layer) == expected
