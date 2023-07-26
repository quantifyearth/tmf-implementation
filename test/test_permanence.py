import math
import pytest
import pandas as pd

from methods.outputs import calculate_permanence # pylint: disable=E0401

test_add = pd.DataFrame([ 4.4, 5.5, 7.7 ], index=[2012, 2013, 2014])
test_leak = pd.DataFrame([ 3.3, 4.4, 5.5 ], index=[2012, 2013, 2014])
test_release_schedule_data = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
]
test_release_schedule = pd.DataFrame(test_release_schedule_data, index=[2012, 2013, 2014], columns=[2012, 2013, 2014])
test_scc = [ 4.0, 4.0, 4.0, 5.0, 5.0, 5.0 ]

def assert_float_eq(value_a, value_b, eps=0.0000001):
    assert((math.isnan(value_a) and math.isnan(value_b)) or (value_a == value_b) or (abs(value_a - value_b) <= eps))

def test_net_sequester():
    calc = calculate_permanence.net_sequestration(test_add, test_leak, 2013)
    assert_float_eq(calc, 0.0)

    # Quick sanity check
    with pytest.raises(ValueError):
        calculate_permanence.net_sequestration(test_add, test_leak, 2012)

def test_release():
    calc = calculate_permanence.release(test_add, test_leak, 2014, 1)
    assert_float_eq(calc, 1.1)

    # Quick sanity check
    with pytest.raises(ValueError):
        calculate_permanence.release(test_add, test_leak, 2014, 3)

def test_adjust():
    expect = calculate_permanence.net_sequestration(test_add, test_leak, 2014) - 2.0
    calc = calculate_permanence.adjusted_net_sequestration(test_add, test_leak, test_release_schedule, 2014)
    assert_float_eq(calc, expect)

    expect = calculate_permanence.net_sequestration(test_add, test_leak, 2013) - 1.0
    calc = calculate_permanence.adjusted_net_sequestration(test_add, test_leak, test_release_schedule, 2013)
    assert_float_eq(calc, expect)

def test_ep_one():
    # In the case where there is (a) no leakage and (b) no release we should come out the
    # other side with essentially a truly permanent carbon credit and thus eP should equal
    # (in a floating point sense) 1.0
    start_year = 2012
    end_year = start_year + 6

    years = range(start_year, end_year)
    # With this additionality and leakage, from year 5 we will get C(5) = 1.0
    add =  pd.DataFrame([ 9.0, 9.0, 9.0, 9.0, 9.0, 10.0 ], index=years)
    leak = pd.DataFrame([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], index=years)

    # We don't actually release any because we are mimicking a permanent credit
    # the 7 for release should really be infinity since there is no release year
    # in this example where all net sequestered carbon is released, contrived but
    # should still work
    release_schedule_data = [
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    ]

    release_schedule = pd.DataFrame(release_schedule_data, index=range(start_year, end_year + 1), columns=years)

    # Some SCC values
    scc = pd.DataFrame([ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0 ], index=range(start_year, end_year + 1), columns=["central"])
    scc = calculate_permanence.interpolate_scc(scc, 2005, 2050)
    expect = 1.0
    calc = calculate_permanence.equivalent_permanence(add, leak, scc, years[5], end_year + 1, release_schedule)
    assert_float_eq(calc, expect)

def test_ep_zero():
    # The opposite of test_ep_one where we release everything, the true impermanent credit
    start_year = 2012
    end_year = start_year + 6

    years = range(start_year, end_year)
    add =  pd.DataFrame([ 9.0, 9.0, 9.0, 9.0, 9.0, 10.0 ], index=years)
    leak = pd.DataFrame([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], index=years)

    # We set the delta value to zero to make the calculations more obvious, that is, we do not
    # discount things in the future in these calculations
    delta = 0.0

    # Same release schedule as above expect in the fifth year estimates we release all of the
    # carbon over a two year period
    release_schedule_data = [
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ]
    ]
    release_schedule = pd.DataFrame(release_schedule_data, index=range(start_year, end_year + 1), columns=years)

    # Some SCC values
    scc = pd.DataFrame([ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0 ], index=range(start_year, end_year + 1), columns=["central"])
    scc = calculate_permanence.interpolate_scc(scc, 2005, 2050)

    expect = 0.0
    calc = calculate_permanence.equivalent_permanence(add, leak, scc, years[5], end_year + 1, release_schedule, delta)
    assert_float_eq(calc, expect)

    # As a sanity check, we'll now keep the default delta value of 3% but increase the SCC
    # for the second year by a similar amount to make sure it is equivalent as our first scenario
    scc = pd.DataFrame([ 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, (500.0 * 1.03) ], index=range(start_year, end_year + 1), columns=["central"])
    scc = calculate_permanence.interpolate_scc(scc, 2005, 2050)
    expect = 0.0
    calc = calculate_permanence.equivalent_permanence(add, leak, scc, years[5], end_year + 1, release_schedule)
    assert_float_eq(calc, expect)

def test_make_release():
    start_year = 2012
    end_year = start_year + 6
    years = range(start_year, end_year + 1)
    now = end_year + 1
    project_end = end_year + 4
    carry_on_to = end_year + 6
    quality = "high"

    additionality = pd.DataFrame([ 800, 400, 200, 100, 50, 25, 12.5 ], index=years)
    leakage = pd.DataFrame([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], index=years)
    schedule = []
    for est in range(start_year, now):
        print(est)
        estimates = []
        for fut in range(start_year, carry_on_to):
            rel_sched = calculate_permanence.release_schedule(quality, additionality, leakage, est, fut, project_end)
            estimates.append(rel_sched)
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
