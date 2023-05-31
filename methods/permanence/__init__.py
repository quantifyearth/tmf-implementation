from typing import Literal, NoReturn, List

def assert_never(value: NoReturn) -> NoReturn:
    assert False,  f'This code should never be reached, got: {value}'

ProjectQuality = Literal["low", "high"]

def net_sequestration(
    additionality : List[float],
    leakage : List[float],
    year : int
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-net-sequestration for a given
    end time i. The additionality and leakage values should be order from oldest
    to newest.

    Args:
        additionality: an ordered list of addtionality values per year for a project
        leakage: an ordered list of leakage values per year for a project
        year: the year to calculate the net sequestration for

    Raises if i is less than 1 or greater than the length of the list and returns
    """
    a_len = len(additionality)
    l_len = len(leakage)

    if a_len != l_len:
        raise ValueError("addaitionality and leakage lists not of equal length")

    if year < 1 or year > a_len:
        raise ValueError(f"index for net sequesteration out of bounds: {year}")

    additionality_t = additionality[year]
    leakage_t = leakage[year]
    additionality_t_prev = additionality[year - 1]
    leakage_t_prev = leakage[year - 1]

    return (additionality_t - leakage_t) - (additionality_t_prev - leakage_t_prev)

def release(
    additionality : List[float],
    leakage : List[float],
    end : int,
    years : int
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-release

    The arguments are similar as to net_sequestration but here we calculate the released
    carbon between [end-years; end].
    """
    a_len = len(additionality)
    l_len = len(leakage)

    if a_len != l_len:
        raise ValueError("additionality and leakage lists not of equal length")

    start = end - years

    if start < 0 or end > a_len:
        raise ValueError(f"end year out of bounds, or too close to the start for given years s: {start} e: {end}")

    net_end = net_sequestration(additionality, leakage, end)
    net_prev = net_sequestration(additionality, leakage, start)

    return (net_end - net_prev) / years

def adjusted_net_sequestration(
    additionality : List[float],
    leakage : List[float],
    schedule : List[List[float]],
    year : int
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-adj

    Additionality and leakage are like those used in net_sequestration.

    The release schedule is a matrix of values indicating the anticipated release
    of carbon for a given years, estimated in another year. The value given by
    release_schedule[i][j] should be the anticipated release of carbon for year
    j as estimated in year i. Any values in the matrix where i >= j will not be used.
    """

    adjustment = 0.0
    for est in range(0, year):
        adjustment += schedule[est][year]

    return net_sequestration(additionality, leakage, year) - adjustment

def release_schedule(
    quality : ProjectQuality,
    additionality : List[float],
    leakage : List[float],
    from_year_index : int,
    to_year_index : int,
    project_end : int
) -> float:
    """
    Implements the anticpated release schedule algorithm ./rfc/permanence/index.html#name-anticipated-release

    Note, this deviates slightly in the RFC section in that we don't return zero when the adjusted net sequestration
    goes to zero. That is up to the end user to check should they compute past this point.
    """
    if quality == 'low':
        return release(additionality, leakage, from_year_index, 5)
    elif quality == 'high':
        # TODO: check strictness
        if to_year_index <= project_end:
            return 0.0
        else:
            # Fiver years plus on for net seq calc
            if from_year_index < 6:
                return 0.0
            else:
                return release(additionality, leakage, from_year_index, 5)
    assert_never(quality)


DEFAULT_DELTA_PER_YEAR = 0.03 # 3% per year

def damage(
    scc : List[float],
    year : int,
    release_year : int,
    schedule : List[List[float]],
    delta : float = DEFAULT_DELTA_PER_YEAR,
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-damage

    Assigns a value to the damage caused by a released amount of carbon in a particular year.

    Args:
        scc: The social cost of carbon values for a year, this list should match that of the release
             schedule i.e., the first value in this list should be for the first year of the project.
        year: The year for which the damage is being calculated.
        release_year: The year by which all of the carbon will have been released by.
        delta: The discount factor.
    """

    damage_acc = 0.0
    years_to_release = release_year - year
    for k in range(0, years_to_release):
        damage_acc += (schedule[year][year + k] * scc[year + k] / (1 + delta) ** k)

    return damage_acc

def equivalent_permanence(
    additionality : List[float],
    leakage : List[float],
    scc : List[float],
    now : int,
    release_year : int,
    schedule : List[List[float]],
    delta : float = DEFAULT_DELTA_PER_YEAR
) -> float:
    adj = adjusted_net_sequestration(additionality, leakage, schedule, now)
    scc_now = scc[now]
    v_adj = adj * scc_now
    return (v_adj - damage(scc, now, release_year, schedule, delta)) / v_adj