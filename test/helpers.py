import json
from typing import List, Set, Tuple

from osgeo import ogr, osr # type: ignore

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

def build_polygon(lat: float, lng: float, radius: float) -> ogr.Geometry:
    frame = {
        'type': 'POLYGON',
        'coordinates': _make_square(lat, lng, radius)
    }
    return ogr.CreateGeometryFromJson(json.dumps(frame))

def build_multipolygon(polygon_list: Set[Tuple[float]]) -> ogr.Geometry:
    frame = {
        'type': 'MULTIPOLYGON',
        'coordinates': [_make_square(*poly) for poly in polygon_list]
    }
    return ogr.CreateGeometryFromJson(json.dumps(frame))

def build_datasource(geometry_list: List[ogr.Geometry]) -> ogr.DataSource:
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326) # aka WSG84

    test_data_source = ogr.GetDriverByName('Memory').CreateDataSource('random name here')
    test_layer = test_data_source.CreateLayer("buffer", spatial_ref, geom_type=ogr.wkbMultiPolygon)
    feature_definition = test_layer.GetLayerDefn()
    for geometry in geometry_list:
        new_feature = ogr.Feature(feature_definition)
        new_feature.SetGeometry(geometry)
        test_layer.CreateFeature(new_feature)

    return test_data_source
