import math
import random
import tempfile
import os

import pandas as pd
import numpy as np

from methods.common.additionality import generate_additionality, MOLECULAR_MASS_CO2_TO_C_RATIO # pylint: disable=E0401

RANDOM_SEED = 59

random.seed(RANDOM_SEED)

def assert_float_eq(value_a, value_b, eps=0.0000001):
    assert (
        (math.isnan(value_a) and math.isnan(value_b))
        or (value_a == value_b)
        or (abs(value_a - value_b) <= eps)
    )

# The LUCs should be a 2-dimensional array. The rows
# are for each counterfactual matching and the columns
# for each year.
def data_from_lucs(k_lucs, s_lucs, start_year):
    years = k_lucs.shape[1]
    data = {}

    k_transpose = np.transpose(k_lucs)
    s_transpose = np.transpose(s_lucs)

    # These are the only fields that need to exist for additionality
    # to work for now.
    for i in range(years):
        print(k_transpose.shape, i)
        year_index = start_year + i
        data[f"k_luc_{year_index}"] = k_transpose[i]
        data[f"s_luc_{year_index}"] = s_transpose[i]

    return data


def test_additionality_all_forest():
    # TODO: Probably produce a schema for this file, at the moment it is
    # explicitly defined anywhere.
    NUMBER_OF_PIXELS = 100

    # All forest, both in the counterfactual and in the control
    k_lucs = np.ones(shape=(NUMBER_OF_PIXELS, 22))
    s_lucs = np.ones(shape=(NUMBER_OF_PIXELS, 22))

    start_year = 2000
    end_year = 2020

    data = data_from_lucs(k_lucs, s_lucs, start_year)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        matches_path = os.path.join(tmpdir, "matches")
        os.mkdir(matches_path)

        df = pd.DataFrame.from_dict(data)
        df.to_parquet(os.path.join(matches_path, "1234.parquet"))

        density = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        area = 10000

        add = generate_additionality(
            area,
            2012,
            end_year,
            density,
            matches_path,
            1
        )

        expected = {}
        for i in range(end_year - start_year + 1):
            expected[start_year + i] = 0.0

        assert(expected == add)

def test_additionality_all_additional():
    # TODO: Probably produce a schema for this file, at the moment it is
    # explicitly defined anywhere.
    NUMBER_OF_PIXELS = 1

    # All deforested in the controls and all forest in the treatment. This means
    # there should be good additionality for each year.
    k_lucs = np.ones(shape=(NUMBER_OF_PIXELS, 22))
    s_lucs = np.ones(shape=(NUMBER_OF_PIXELS, 22)) + 2

    start_year = 2000
    end_year = 2020

    data = data_from_lucs(k_lucs, s_lucs, start_year)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        matches_path = os.path.join(tmpdir, "matches")
        os.mkdir(matches_path)

        df = pd.DataFrame.from_dict(data)
        df.to_parquet(os.path.join(matches_path, "1234.parquet"))

        density = np.array([100.0, 50.0, 10.0, 0.0, 0.0, 0.0])
        area = 10000

        add = generate_additionality(
            area,
            2012,
            end_year,
            density,
            matches_path,
            1
        )

        # We make the calculation for expected easier by removing
        # this scaling factor.
        for k, v in add.items():
            add[k] = v / MOLECULAR_MASS_CO2_TO_C_RATIO

        # We expect 90 here as our carbon density for undisturbed is 100
        # and for deforested it is 10.
        for i in range(end_year - start_year + 1):
            assert_float_eq(add[start_year + i], 90.0)
