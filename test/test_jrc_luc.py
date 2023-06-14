import pytest

from methods.inputs.generate_luc_layer import directed_to_regular # pylint: disable=E0401

@pytest.mark.parametrize(
    "value,expected",
    [
        ("N10", 10),
        ("S10", -10),
        ("E10", 10),
        ("W10", -10),
        ("N0", 0),
        ("S0", 0),
        ("E0", 0),
        ("W0", 0),
    ]
)
def test_unmunge_direction(value, expected):
    result = directed_to_regular(value)
    assert result == expected
