from typing import Set, Tuple

import pytest
from geopandas import gpd

from methods.common.geometry import utm_for_geometry, expand_boundaries
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
    utm_code = utm_for_geometry(lat, lng)
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
    test_gdf = gpd.GeoDataFrame.from_features(gpd.GeoSeries(test_poly), crs="EPSG:4326")
    original_area = test_gdf.to_crs('3857').area.sum()

    expanded_gdf = expand_boundaries(test_gdf, 1000)
    expanded_area = expanded_gdf.to_crs('3857').area.sum()

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
    test_polygons = [build_polygon(*poly) for poly in polygon_list]
    test_gdf = gpd.GeoDataFrame.from_features(gpd.GeoSeries(test_polygons), crs="EPSG:4326")

    expanded = expand_boundaries(test_gdf, 1000)
    assert expanded.shape[0] == 1

    # because of the user of unary_uniform, we expect there to be one multipolygon geometry,
    # or one polygon
    assert len(expanded.geometry) == 1

    top_level = expanded.geometry[0]
    if expected_count > 1:
        assert top_level.geom_type == 'MultiPolygon'
        assert len(top_level.geoms) == expected_count
    else:
        assert top_level.geom_type == 'Polygon'
