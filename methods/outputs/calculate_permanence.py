import json
import argparse
import logging
from typing import Literal, NoReturn

import pandas as pd # type: ignore
import numpy as np # type: ignore
from scipy.interpolate import interp1d # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def assert_never(value: NoReturn) -> NoReturn:
    assert False, f"This code should never be reached, got: {value}"


ProjectQuality = Literal["low", "high"]


def net_sequestration(
    additionality: pd.DataFrame, leakage: pd.DataFrame, year: int
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
        raise ValueError("additionality and leakage lists not of equal length")

    minimum_year = int(additionality.index.min())
    maximum_year = int(additionality.index.max())

    if year < minimum_year + 1 or year > maximum_year:
        raise ValueError(f"index for net sequesteration out of bounds: {year}")

    additionality_t = float(additionality.loc[year][0])
    leakage_t = float(leakage.loc[year][0])
    additionality_t_prev = float(additionality.loc[year - 1][0])
    leakage_t_prev = float(leakage.loc[year - 1][0])

    return (additionality_t - leakage_t) - (additionality_t_prev - leakage_t_prev)


def release(
    additionality: pd.DataFrame, leakage: pd.DataFrame, end_year: int, years: int
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

    start_year = end_year - years
    minimum_year = int(additionality.index.min())
    maximum_year = int(additionality.index.max())

    if start_year < minimum_year or end_year > maximum_year:
        raise ValueError(
            f"end year out of bounds, or too close to the start \
              for given years start: {start_year}, max: {maximum_year}, end: {end_year}"
        )

    net_end = net_sequestration(additionality, leakage, end_year)
    net_prev = net_sequestration(additionality, leakage, start_year)

    return (net_end - net_prev) / years


def adjusted_net_sequestration(
    additionality: pd.DataFrame,
    leakage: pd.DataFrame,
    schedule: pd.DataFrame,
    year: int,
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
    minimum_year = int(schedule.index.min())

    for est_year in range(minimum_year, year):
        estimate = schedule[est_year]
        adjustment += estimate[year]

    return net_sequestration(additionality, leakage, year) - adjustment


def release_schedule(
    quality: ProjectQuality,
    additionality: pd.DataFrame,
    leakage: pd.DataFrame,
    from_year: int,
    to_year: int,
    project_end: int,
) -> float:
    """
    Implements the anticpated release schedule algorithm ./rfc/permanence/index.html#name-anticipated-release

    Note, this deviates slightly in the RFC section in that we don't return zero when the adjusted net sequestration
    goes to zero. That is up to the end user to check should they compute past this point.
    """
    minimum_year = int(additionality.index.min())

    if quality == "low":  # i.e. HIGH RISK
        return release(additionality, leakage, from_year, 5)
    elif quality == "high":  # i.e. LOW RISK
        # TODO: check strictness
        if to_year <= project_end:
            return 0.0
        else:
            # Five years plus one for net seq calc
            if from_year < minimum_year + 6:
                return 0.0
            else:
                return release(additionality, leakage, from_year, 5)
    assert_never(quality)


DEFAULT_DELTA_PER_YEAR = 0.03  # 3% per year


def damage(
    scc: pd.DataFrame,
    year: int,
    release_year: int,
    schedule: pd.DataFrame,
    delta: float = DEFAULT_DELTA_PER_YEAR,
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-damage

    Assigns a value to the damage caused by a released amount of carbon in a particular year.

    Args:
        scc: The social cost of carbon values for a year, this list should match that of the release
             schedule i.e., the first value in this list should be for the first year of the project.
        year_idx: The year for which the damage is being calculated.
        release_year_idx: The year by which all of the carbon will have been released by. Should be
                      greater than 'year'.
        schedule: The release schedule for the project, this should be a rectangularly-shaped matrix of
                  values.
        delta: The discount factor.
    """
    if release_year <= year:
        raise ValueError(
            "The release year must be greater than the year damage is being calculated for"
        )

    years_to_release = release_year - year

    sched_estimate_min_year = schedule.index.min()
    sched_estimate_max_year = schedule.index.max()

    if sched_estimate_max_year < year:
        raise ValueError(
            f"The release schedule does not make estimates for the year {year}"
        )

    sched_years_max = sched_estimate_min_year + len(schedule[sched_estimate_min_year])
    scc_year_max = scc.index.max()
    maximum_forecast = year + years_to_release

    if sched_years_max < maximum_forecast:
        raise ValueError(
            f"""The release schedule should contain
        anticipated releases up to the year indexed by {maximum_forecast}"""
        )

    if scc_year_max < maximum_forecast:
        raise ValueError(
            f"""Not enough values were provided for the Social Cost of Carbon,
        only {scc_year_max} were given and we need {maximum_forecast}"""
        )

    damage_acc = 0.0

    for k in range(0, years_to_release):
        release_amt = schedule[year][year + k]
        carbon = scc.loc[year + k][0]
        damage_acc += abs(release_amt) * carbon / ((1 + delta) ** k)

    return damage_acc


def equivalent_permanence(
    additionality: pd.DataFrame,
    leakage: pd.DataFrame,
    scc: pd.DataFrame,
    current_year: int,
    schedule: pd.DataFrame,
    delta: float = DEFAULT_DELTA_PER_YEAR,
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-ep

    Calculates the equivalent permanence.

    Args:
      additionality: Values for the additionality in the project for each evaluation year. 
                     One column for year another for additionality.
      leakage: Values for the leakage in the project for each evaluation year, like additionality.
      scc: Values for the Social Cost of Carbon for each year.
      current_year: The current evaluation year.
      release_year: The year by which all net sequestration has been released.
      schedule: A release schedule for the project, see the function release_schedule for how you might
                wish to calculate this.
      delta: A parameter used to discount the damage into the future, this defaults to 0.03 (3%).
    """
    add_len = len(additionality)
    leak_len = len(leakage)

    if add_len != leak_len:
        raise ValueError(
            "The number of values for additionality and leakage are not the same"
        )

    release_yr = current_year

    # TODO: This needs double checking just to be sure it is the right method
    adj = adjusted_net_sequestration(additionality, leakage, schedule, current_year)
    release_amount = 0
    while adj - release_amount > 0:
        release_yr += 1
        release_amount += abs(schedule[current_year][release_yr])

    scc = interpolate_scc(scc, 2005, release_yr)
    scc_now = scc.at[current_year, "value"]

    v_adj = adj * scc_now

    dmg = damage(scc, current_year, release_yr, schedule, delta)
    logging.info(
        "Release year: %i, Damage: %f and Adjusted Net Seq. %f, eP: %f",
        release_yr,
        dmg,
        v_adj,
        (v_adj - dmg) / v_adj,
    )

    return (v_adj - dmg) / v_adj


def interpolate_scc(scc: pd.DataFrame, minimum_year: int, max_year: int) -> pd.DataFrame:
    years = scc.index.tolist()
    values = scc["central"].values
    # TODO: interp1d a fair enough extrapolation technique?
    interpolated = interp1d(years, values, fill_value="extrapolate")
    new_years = np.arange(minimum_year, max_year + 1, 1)
    interp_data = list(zip(new_years, interpolated(new_years)))
    return pd.DataFrame(interp_data, columns=["year", "value"]).set_index("year")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes equivalent permanence from SCC values, additionality and leakage."
    )
    parser.add_argument(
        "--additionality",
        type=str,
        required=True,
        dest="additionality_csv",
        help="A CSV containing additionality for a range of years",
    )
    parser.add_argument(
        "--leakage",
        type=str,
        required=True,
        dest="leakage_csv",
        help="A CSV containing leakage values for the same range of years as additionality",
    )
    parser.add_argument(
        "--scc",
        type=str,
        required=True,
        dest="scc_csv",
        help="A CSV containing social cost of carbon values",
    )
    parser.add_argument(
        "--current_year",
        type=int,
        required=True,
        dest="current_year",
        help="Current year",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_json",
        help="The destination output JSON path.",
    )

    args = parser.parse_args()

    additionality_data = pd.read_csv(args.additionality_csv, index_col="year")
    leakage_data = pd.read_csv(args.leakage_csv, index_col="year")
    scc_data = pd.read_csv(args.scc_csv, index_col="year")

    min_year = additionality_data.index.min()

    schedule_data = []
    for fut in range(min_year, 4000):
        estimates = [float(fut)]
        for est in range(min_year, args.current_year + 1):
            rel_sched = release_schedule("high", additionality_data, leakage_data, est, fut, 2042)
            estimates.append(rel_sched)
        schedule_data.append(estimates)

    columns = ["year"] + list(range(min_year, args.current_year + 1))

    # The schedule dataframe can be accessed as schedule_df[est_year][for_year] i.e.
    # the scheduled release in for_year as estimated in est_year.
    schedule_df_raw = pd.DataFrame(schedule_data, columns=columns)
    SCHEDULE_DF = schedule_df_raw.set_index("year")

    ep = equivalent_permanence(
        additionality_data,
        leakage_data,
        scc_data,
        args.current_year,
        SCHEDULE_DF,
    )

    # TODO: Probably return more than just ep
    with open(args.output_json, "w", encoding="utf-8") as f:
        data = {"ep": ep}
        json.dump(data, f)
