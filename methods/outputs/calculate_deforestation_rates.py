import argparse
import glob
import os

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from methods.common.additionality import find_first_luc, is_not_matchless
from methods.common import LandUseClass


def deforestation_rates(matches_directory: str, end_year: int) -> pd.DataFrame:
    matches = glob.glob("*.parquet", root_dir=matches_directory)
    matches = list(filter(is_not_matchless, matches))

    rates_for_matches = []

    for pairs in matches:
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        columns = matches_df.columns.to_list()
        columns.sort()

        earliest_year = find_first_luc(columns)

        total_pixels = len(matches_df)
        rates = []

        for year_index in range(earliest_year, end_year):
            project_deforested = matches_df[f"k_luc_{year_index}"].value_counts()[
                LandUseClass.DEFORESTED
            ]
            project_deforested_next = matches_df[
                f"k_luc_{year_index + 1}"
            ].value_counts()[LandUseClass.DEFORESTED]
            rates.append(np.array([project_deforested, project_deforested_next]))

        rates_for_matches.append(rates)

    average_deforested_pixels = np.sum(rates_for_matches, axis=0) / len(
        rates_for_matches
    )

    results = []

    # We now add extra data (like the years into the data) and compute the percentage
    start_year = earliest_year
    for pair in average_deforested_pixels:
        results.append(
            [
                start_year,
                start_year + 1,
                pair[0],
                pair[1],
                ((pair[1] - pair[0]) / total_pixels) * 100,
            ]
        )

        start_year += 1

    output = pd.DataFrame.from_records(
        results,
        columns=[
            "start year",
            "end year",
            "average number of deforested pixels (start year)",
            "average number of deforested pixels (end year)",
            "deforestation rate (percentage)",
        ],
    )

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the yearly deforestation rate for a treatment area and the control."
    )
    parser.add_argument(
        "--matches",
        type=str,
        required=True,
        dest="matches_dir",
        help="A directory containing counterfactual pixel matches",
    )

    parser.add_argument(
        "--end-year",
        type=str,
        required=True,
        dest="end_year",
        help="The final year to include in the deforestation rate calculations",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output",
        help="The output CSV file name",
    )

    args = parser.parse_args()

    def_rates = deforestation_rates(args.matches_dir, args.end_year)

    def_rates.to_csv(args.output)
