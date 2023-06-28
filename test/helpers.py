from typing import List

from shapely.geometry import Polygon

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

def build_polygon(lat: float, lng: float, radius: float) -> Polygon:
    return Polygon(_make_square(lat, lng, radius)[0])
