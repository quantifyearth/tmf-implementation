from methods import permanence
import pytest
import math
import sys

test_add = [ 4.4, 5.5, 7.7 ]
test_leak = [ 3.3, 4.4, 5.5 ]

def assert_float_eq(a, b, eps=0.0000001):
    assert((math.isnan(a) and math.isnan(b)) or (a == b) or (abs(a - b) <= eps))

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