import math

import numpy as np
import scipy # type: ignore
import shapely # type: ignore
from geopandas import gpd # type: ignore

def utm_for_geometry(lat: float, lng: float) -> int:
    utm_band = (math.floor((lng + 180.0) / 6.0) % 60) + 1
    if lat < 0.0:
        epsg_code = 32700 + utm_band
    else:
        epsg_code = 32600 + utm_band
    return epsg_code

def expand_boundaries(source: gpd.GeoDataFrame, radius: float) -> gpd.GeoDataFrame:
    original_projection = source.crs

    utm_codes = np.array([utm_for_geometry(x.centroid.y, x.centroid.x) for x in source.geometry])
    utm_code = scipy.stats.mode(utm_codes, keepdims=False).mode
    projected_boundaries = source.to_crs(f"EPSG:{utm_code}")

    expanded = [x.buffer(radius) for x in projected_boundaries.geometry]
    simplified = shapely.unary_union(expanded)

    finished = gpd.GeoSeries(simplified)
    utm_result = gpd.GeoDataFrame.from_features(finished, crs=f"EPSG:{utm_code}")
    return utm_result.to_crs(original_projection)
