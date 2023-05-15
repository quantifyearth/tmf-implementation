def net_sequestration(
    additionality : list[float],
    leakage : list[float],
    i : int
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-net-sequestration for a given
    end time i. The additionality and leakage values should be order from oldest
    to newest.

    Raises if i is less than 1 or greater than the length of the list.
    """
    a_len = len(additionality)
    l_len = len(leakage)

    if a_len != l_len:
        raise ValueError("addaitionality and leakage lists not of equal length")

    if i < 1 or i > a_len:
        raise ValueError("index for net sequesteration out of bounds")

    additionality_t = additionality[i]
    leakage_t = leakage[i]
    additionality_t_prev = additionality[i - 1]
    leakage_t_prev = leakage[i - 1]

    return (additionality_t - leakage_t) - (additionality_t_prev - leakage_t_prev)

def release(
    additionality : list[float],
    leakage : list[float],
    end : int,
    years : int
) -> float:
    """
    Implements ./rfc/permanence/index.html#name-release
    """
    a_len = len(additionality)
    l_len = len(leakage)

    if a_len != l_len:
        raise ValueError("additionality and leakage lists not of equal length")

    start = end - years

    if start < 0 or end > a_len:
        raise ValueError("end year out of bounds, or to close to the start for given years")

    net_end = net_sequestration(additionality, leakage, end)
    net_prev = net_sequestration(additionality, leakage, start)

    return (net_end - net_prev) / years
