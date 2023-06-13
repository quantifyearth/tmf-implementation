import math

import pytest

from methods import permanence # pylint: disable=E0401

test_add = [ 4.4, 5.5, 7.7 ]
test_leak = [ 3.3, 4.4, 5.5 ]

def assert_float_eq(value_a, value_b, eps=0.0000001):
    assert((math.isnan(value_a) and math.isnan(value_b)) or (value_a == value_b) or (abs(value_a - value_b) <= eps))

def test_net_sequester():
    calc = permanence.net_sequestration(test_add, test_leak, 1)
    assert_float_eq(calc, 0.0)

    # Quick sanity check
    with pytest.raises(ValueError):
        permanence.net_sequestration(test_add, test_leak, 0)

def test_release():
    calc = permanence.release(test_add, test_leak, 2, 1)
    assert_float_eq(calc, 1.1)

    # Quick sanity check
    with pytest.raises(ValueError):
        permanence.release(test_add, test_leak, 2, 3)