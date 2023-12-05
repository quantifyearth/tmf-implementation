import glob
import os
import logging
from typing import Dict, Any, List

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from geojson import LineString, FeatureCollection, Feature, MultiPoint, dumps  # type: ignore

from methods.common import LandUseClass, partials_dir

MOLECULAR_MASS_CO2_TO_C_RATIO = 44 / 12

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def plot_carbon_stock(
    axis: plt.Axes,
    project_data: Dict[int, float],
    control_data: Dict[int, float],
    start_year: int
) -> None:
    """Will plot the carbon stock for a project and the controls. Those dictionaries should
    be the yearly carbon stock."""
    x_axis = []
    treatment = []
    control = []
    for year, value in project_data.items():
        x_axis.append(year)
        treatment.append(value)
        control.append(control_data[year])
    axis.plot(x_axis, treatment, label="Treatment")
    axis.plot(x_axis, control, label="Control")
    axis.set_title("Carbon stock (Average Treatment and Average Control)")
    axis.set_xlabel("Year")
    axis.set_ylabel("Carbon Stock (MgCO2e)")
    axis.axvline(start_year)
    axis.legend(loc="lower left")


def plot_carbon_trajectories(
    axis: List[plt.Axes],
    title: str,
    idx: int,
    timeseries: Dict[int, np.ndarray],
    start_year: str
):
    x_axis = []
    y_axis = []
    for year, value in timeseries.items():
        x_axis.append(year)
        y_axis.append(value)
    axis[idx].plot(x_axis, y_axis)
    axis[idx].set_title(title)
    axis[idx].set_xlabel("Year")
    axis[idx].set_ylabel("Carbon Stock (MgCO2e)")
    axis[idx].axvline(int(start_year))


def find_first_luc(columns: list[str]) -> int:
    for col in columns:
        split = col.split("_luc_")
        if len(split) < 2:
            continue
        try:
            return int(split[1])
        except ValueError:
            continue
    raise ValueError("Failed to extract earliest year from LUCs")


def is_not_matchless(path: str) -> bool:
    return not path.endswith("_matchless.parquet")


def generate_additionality(
    project_area_msq: float,
    project_start: str,
    end_year: int,
    density: np.ndarray,
    matches_directory: str,
    expected_number_of_iterations: int,
) -> Dict[int, float]:
    """Calculate the additionality (or leakage) of a project from the counterfactual pair matchings
    alongside the carbon density values and some project specific metadata."""
    logging.info("Project area: %.2fmsq", project_area_msq)

    matches = glob.glob("*.parquet", root_dir=matches_directory)
    matches = [x for x in matches if is_not_matchless(x)]
    assert len(matches) == expected_number_of_iterations

    treatment_data : Dict[int, np.ndarray] = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing additionality in treatment for %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        columns = matches_df.columns.to_list()
        columns.sort()

        earliest_year = find_first_luc(columns)

        for year_index in range(earliest_year, end_year + 1):
            total_pixels_t = len(matches_df)

            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"k_luc_{year_index}"].value_counts()

            for luc in LandUseClass:
                if value_count_year.get(luc.value) is not None:
                    values[luc.value - 1] = value_count_year[luc.value]

            undisturbed_t = values[LandUseClass.UNDISTURBED - 1]
            degraded_t = values[LandUseClass.DEGRADED - 1]
            deforested_t = values[LandUseClass.DEFORESTED - 1]
            regrowth_t = values[LandUseClass.REGROWTH - 1]
            water_t = values[LandUseClass.WATER - 1]
            other_t = values[LandUseClass.OTHER - 1]

            proportions_t = np.array(
                [
                    undisturbed_t / total_pixels_t,
                    degraded_t / total_pixels_t,
                    deforested_t / total_pixels_t,
                    regrowth_t / total_pixels_t,
                    water_t / total_pixels_t,
                    other_t / total_pixels_t,
                ]
            )

            # Quick Sanity Check
            prop = np.sum(proportions_t)
            assert 0.99 < prop < 1.01

            areas_t = proportions_t * (project_area_msq / 10000)
            s_t = areas_t * density

            s_t_value = s_t.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            logging.info("Additionality in treatment is %f", s_t_value)

            if treatment_data.get(year_index) is not None:
                treatment_data[year_index][pair_idx] = s_t_value
            else:
                arr = np.zeros(expected_number_of_iterations)
                arr[pair_idx] = s_t_value
                treatment_data[year_index] = arr

    scvt : Dict[int, np.ndarray] = {}

    for pair_idx, pairs in enumerate(matches):
        logging.info("Computing additionality for control %s", pairs)
        matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

        columns = matches_df.columns.to_list()
        columns.sort()

        earliest_year = find_first_luc(columns)

        if earliest_year is None:
            raise ValueError("Failed to extract earliest year from LUCs")

        total_pixels_c = len(matches_df)
        for year_index in range(earliest_year, end_year + 1):
            values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            value_count_year = matches_df[f"s_luc_{year_index}"].value_counts()

            for luc in LandUseClass:
                if value_count_year.get(luc.value) is not None:
                    values[luc.value - 1] = value_count_year[luc.value]

            undisturbed_c = values[LandUseClass.UNDISTURBED - 1]
            degraded_c = values[LandUseClass.DEGRADED - 1]
            deforested_c = values[LandUseClass.DEFORESTED - 1]
            regrowth_c = values[LandUseClass.REGROWTH - 1]
            water_c = values[LandUseClass.WATER - 1]
            other_c = values[LandUseClass.OTHER - 1]

            proportions_c = np.array(
                [
                    undisturbed_c / total_pixels_c,
                    degraded_c / total_pixels_c,
                    deforested_c / total_pixels_c,
                    regrowth_c / total_pixels_c,
                    water_c / total_pixels_c,
                    other_c / total_pixels_c,
                ]
            )

            # Quick Sanity Check
            prop = np.sum(proportions_c)
            assert 0.99 < prop < 1.01

            areas_c = proportions_c * (project_area_msq / 10000)
            s_c = areas_c * density

            s_c_value = s_c.sum() * MOLECULAR_MASS_CO2_TO_C_RATIO

            logging.info("Additionality in counterfactual is %f", s_c_value)

            if scvt.get(year_index) is not None:
                scvt[year_index][pair_idx] = s_c_value
            else:
                arr = np.zeros(expected_number_of_iterations)
                arr[pair_idx] = s_c_value
                scvt[year_index] = arr

    c_tot : Dict[int, float] = {}
    for year, values in scvt.items():
        c_tot[year] = np.average(values)

    p_tot : Dict[int, float] = {}
    for year, values in treatment_data.items():
        p_tot[year] = np.average(values)

    if partials_dir is not None:
        figure, axis = plt.subplots(1, 3)
        figure.set_figheight(10)
        figure.set_figwidth(18)

        plot_carbon_trajectories(
            axis,
            "Carbon stock (All Matches Treatment)",
            1,
            treatment_data,
            project_start,
        )
        plot_carbon_trajectories(
            axis, "Carbon stock (All Matches Control)", 2, scvt, project_start
        )
        plot_carbon_stock(axis[0], p_tot, c_tot, int(project_start))

        out_path = os.path.join(
            partials_dir, os.path.splitext(pairs)[0] + "-carbon-stock.png"
        )

        figure.savefig(out_path)

        # Now for all the pairs we create a GeoJSON for visualising
        smds : Dict[str, Any] = {"pair_id": [], "feature": [], "smd": []}
        for pair_idx, pairs in enumerate(matches):
            matches_df = pd.read_parquet(os.path.join(matches_directory, pairs))

            linestrings = []
            for _, row in matches_df.iterrows():
                linestring = Feature(
                    geometry=LineString(
                        [(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]
                    )
                )
                linestrings.append(linestring)

            geomtry_collection = FeatureCollection(linestrings)
            out_path = os.path.join(
                partials_dir, os.path.splitext(pairs)[0] + "-pairs.geojson"
            )

            with open(out_path, "w", encoding="utf-8") as output_file:
                output_file.write(dumps(geomtry_collection))

            points = []
            for _, row in matches_df.iterrows():
                linestring = Feature(
                    geometry=MultiPoint(
                        [(row["k_lng"], row["k_lat"]), (row["s_lng"], row["s_lat"])]
                    )
                )
                points.append(linestring)

            points_gc = FeatureCollection(points)
            out_path = os.path.join(
                partials_dir, os.path.splitext(pairs)[0] + "-pairs-points.geojson"
            )

            with open(out_path, "w", encoding="utf-8") as output_file:
                output_file.write(dumps(points_gc))

            # We now compute statistics for each pairing mainly looking at SMD
            mean_std = matches_df.agg(["mean", "std"])
            for col in matches_df.columns:
                # only go from K to S so we don't double count
                if col[0] == "k":
                    treat_mean = mean_std[col]["mean"]
                    feature = "_".join(col.split("_")[1:])
                    control_col = "s_" + feature
                    control_mean = mean_std[control_col]["mean"]
                    treat_std = mean_std[col]["std"]
                    control_std = mean_std[control_col]["std"]
                    smd = (treat_mean - control_mean) / np.sqrt(
                        (treat_std**2 + control_std**2) / 2
                    )
                    smd = round(abs(smd), 8)
                    smds["pair_id"].append(os.path.splitext(pairs)[0])
                    smds["feature"].append(feature)
                    smds["smd"].append(smd)
        smd_path = os.path.join(
            partials_dir,
            "smd.csv"
        )
        smds_df = pd.DataFrame.from_dict(smds)
        smds_df.to_csv(smd_path)

    result : Dict[int, float] = {}

    for year, value in p_tot.items():
        result[year] = value - c_tot[year]

    return result
    