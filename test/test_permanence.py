import math

import pytest

from methods import permanence # pylint: disable=E0401

test_add = [ 4.4, 5.5, 7.7 ]
test_leak = [ 3.3, 4.4, 5.5 ]
test_release_schedule = [
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
]
test_scc = [ 4.0, 4.0, 4.0, 5.0, 5.0, 5.0 ]

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

def test_adjust():
    expect = permanence.net_sequestration(test_add, test_leak, 2) - 2.0
    calc = permanence.adjusted_net_sequestration(test_add, test_leak, test_release_schedule, 2)
    assert_float_eq(calc, expect)

    expect = permanence.net_sequestration(test_add, test_leak, 1) - 1.0
    calc = permanence.adjusted_net_sequestration(test_add, test_leak, test_release_schedule, 1)
    assert_float_eq(calc, expect)

def test_ep_one():
    # In the case where there is (a) no leakage and (b) no release we should come out the
    # other side with essentially a truly permanent carbon credit and thus eP should equal
    # (in a floating point sense) 1.0

    now = 5
    release = 7

    # With this additionality and leakage, from year 5 we will get C(5) = 1.0
    add =  [ 9.0, 9.0, 9.0, 9.0, 9.0, 10.0 ]
    leak = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]

    # We don't actually release any because we are mimicking a permanent credit
    # the 7 for release should really be infinity since there is no release year
    # in this example where all net sequestered carbon is released, contrived but
    # should still work
    release_schedule = [
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    ]

    # Some SCC values
    scc = [ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0 ]
    expect = 1.0
    calc = permanence.equivalent_permanence(add, leak, scc, 5, 7, release_schedule)
    assert_float_eq(calc, expect)

def test_ep_zero():
    # The opposite of test_ep_one where we release everything, the true impermanent credit
    add =  [ 9.0, 9.0, 9.0, 9.0, 9.0, 10.0 ]
    leak = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]

    # We set the delta value to zero to make the calculations more obvious, that is, we do not
    # discount things in the future in these calculations
    delta = 0.0

    # Same release schedule as above expect in the fifth year estimates we release all of the
    # carbon over a two year period
    release_schedule = [
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5 ],
    ]
    scc = [ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0 ]
    expect = 0.0
    calc = permanence.equivalent_permanence(add, leak, scc, 5, 7, release_schedule, delta)
    assert_float_eq(calc, expect)

    # As a sanity check, we'll now keep the default delta value of 3% but increase the SCC
    # for the second year by a similar amount to make sure it is equivalent as our first scenario
    scc = [ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, (500.0 * 1.03) ]
    expect = 0.0
    calc = permanence.equivalent_permanence(add, leak, scc, 5, 7, release_schedule)
    assert_float_eq(calc, expect)

def test_make_release():
    now = 7
    project_end = 10
    carry_on_to = 12
    quality = "high"
    additionality = [ 800, 400, 200, 100, 50, 25, 12.5 ]
    leakage = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    schedule = []
    for est in range(0, now):
        estimates = []
        for fut in range(0, carry_on_to):
            r = permanence.release_schedule(quality, additionality, leakage, est, fut, project_end)
            estimates.append(r)
        schedule.append(estimates)
    # What do we expect?
    # It's high quality, so everything until the project_end is 0.0 estimated from any previous time period
    # Only for values estimated after year 6 (so the averaging calculations work...) should not be 0.0
    # and only values that after the project_end year. So the first value will have to be (7, 11) i.e. in the
    # bottom right corner. And what should the value be?

    # From the RFC, the release will be R = (C(t_i) - C(t_{i - 5})) / 5. To make things simple we only have
    # additionality in this test, and so we get the seventh NET additionality value which is -12.5 and five years
    # before which is -400. This gives us a value of -12.5 - -400 = 387.5. When averaged over five years this
    # is 77.5
    expect = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # TODO: we need an assert_list_float_equal
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.5]
    ]
    assert expect == schedule
