import pytest

from methods.common.geometry import wgs_aspect_ratio_at

def close_enough(a_value, b_value):
    return abs(a_value - b_value) < 0.001

@pytest.mark.parametrize(
    "longitude,aspect_ratio",
    [
        (0, 0.993),
        (38, 69 / 54.6), # https://www.usgs.gov/faqs/how-much-distance-does-a-degree-minute-and-second-cover-your-maps
        (60, 1.996),
    ]
)
def test_wgs_aspect_ratio_at(longitude: float, aspect_ratio: float):
    assert close_enough(aspect_ratio, wgs_aspect_ratio_at(longitude)), f"""
        Aspect ratio at {longitude} degrees should be {aspect_ratio}, got {wgs_aspect_ratio_at(longitude)}"""

    assert close_enough(aspect_ratio, wgs_aspect_ratio_at(-longitude)), f"""
        Aspect ratio at -{longitude} degrees should be {aspect_ratio}, got {wgs_aspect_ratio_at(-longitude)}"""
