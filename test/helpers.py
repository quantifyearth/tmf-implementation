import json

from osgeo import ogr # type: ignore

def build_polygon(lat: float, lng: float, radius: float) -> ogr.Geometry:
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
