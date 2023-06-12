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
    end time i. The additionality and leakage values should be ordered from oldest
    to newest.

    Args:
        additionality: an ordered list of addtionality values per year for a project
        leakage: an ordered list of leakage values per year for a project
        year: the year index to calculate the net sequestration for

    Raises if i is less than 1 or greater than the length of the list.
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
    year_idx : int
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
    for est in range(0, year_idx):
        adjustment += schedule[est][year_idx]

    return net_sequestration(additionality, leakage, year_idx) - adjustment

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
    year_idx : int,
    release_year_idx : int,
    schedule : List[List[float]],
    delta : float = DEFAULT_DELTA_PER_YEAR,
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-damage

    Assigns a value to the damage caused by a released amount of carbon in a particular year.

    Args:
        scc: The social cost of carbon values for a year, this list should match that of the release
             schedule i.e., the first value in this list should be for the first year of the project.
        year_idx: The year index for which the damage is being calculated.
        release_year_idx: The year index by which all of the carbon will have been released by. Should be
                      greater than 'year'.
        schedule: The release schedule for the project, this should be a rectangularly-shaped matrix of
                  values.
        delta: The discount factor.
    """
    if year_idx < 0:
        raise ValueError("Damage year must be greater than or equal to zero")

    if release_year_idx <= year_idx:
        raise ValueError("The release year must be greater than the year damage is being calculated for")

    damage_acc = 0.0
    years_to_release = release_year_idx - year_idx

    sched_estimate_len = len(schedule)

    if sched_estimate_len < year_idx:
        raise ValueError(f"The release schedule does not make estimates in the year indexed by {year_idx}")

    sched_years_len = len(schedule[0])
    scc_len = len(scc)
    maximum_forecast = year_idx + years_to_release

    if sched_years_len < maximum_forecast:
        raise ValueError(f"""The release schedule should contain
        anticipated releases up to the year indexed by {maximum_forecast}""")

    if scc_len < maximum_forecast:
        raise ValueError(f"""Not enough values were provided for the Social Cost of Carbon,
        only {scc_len} were given and we need {maximum_forecast}""")


    for k in range(0, years_to_release):
        damage_acc += (schedule[year_idx][year_idx + k] * scc[year_idx + k] / (1 + delta) ** k)

    return damage_acc

def equivalent_permanence(
    additionality : List[float],
    leakage : List[float],
    scc : List[float],
    now_idx : int,
    release_year_idx : int,
    schedule : List[List[float]],
    delta : float = DEFAULT_DELTA_PER_YEAR
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-ep

    Calculates the equivalent permanence.

    Args:
      additionality: Values for the additionality in the project for each evaluation year
      leakage: Values for the leakage in the project for each evaluation year
      scc: Values for the Social Cost of Carbon for each year. This should be indexed in the same
           way as additionality and leakage. For example, the first value in additionality is the
           amount of additionality in the first year of the project which may correspond to the year
           2000. The first value in SCC should therefore be the SCC in the year 2000.
      now_idx: The current evaluation year (as an index into the additionality and leakage values).
      release_year_idx: The year by which all net sequestration has been released, again given as an index.
      schedule: A release schedule for the project, see the function release_schedule for how you might
                wish to calculate this.
      delta: A parameter used to discount the damage into the future, this defaults to 0.03 (3%).
    """
    add_len = len(additionality)
    leak_len = len(leakage)
    scc_len = len(scc)

    if add_len != leak_len:
        raise ValueError("The number of values for additionality and leakage are not the same")

    if now_idx >= scc_len:
        raise ValueError(f"No SCC value for the year indexed by {now_idx}")

    scc_now = scc[now_idx]
    adj = adjusted_net_sequestration(additionality, leakage, schedule, now_idx)
    v_adj = adj * scc_now
    return (v_adj - damage(scc, now_idx, release_year_idx, schedule, delta)) / v_adj