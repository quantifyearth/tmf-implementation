import json
import argparse
import logging
import sys # Import sys for exit
from typing import Literal, NoReturn, Optional # Added Optional

import pandas as pd # type: ignore
import numpy as np # type: ignore
from scipy.interpolate import interp1d # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- additionality_rate function (remains the same) ---
def additionality_rate(
    additionality_data: pd.DataFrame):
    """
    Computes a rate of change based on recent additionality data.
    Uses last 5 years if available, otherwise averages from year 12 onwards.
    Assumes index 'year' exists and data starts potentially before year 12.
    """
    # Ensure we only use data up to the last available year for rate calculation
    if len(additionality_data) == 0:
         return 0.0 # No data, no rate

    # Find the index corresponding to the 12th year (index 11) if it exists
    start_calc_year_index = 11 # Define index for 12th row
    if len(additionality_data) <= start_calc_year_index:
        logging.warning("Not enough data (<= 11 years) to calculate rate reliably. Returning 0.")
        return 0.0

    # Check if we have at least 5 years *after* the 11th year (i.e., >= 16 years total)
    if len(additionality_data) >= start_calc_year_index + 5:
        # Use the average of the last 5 years relative to the end
        rate = (
            additionality_data['additionality_mean'].iloc[-1] - additionality_data['additionality_mean'].iloc[-6]
        ) / 5
        logging.debug(f"Calculated 5-year rate: {rate}")
    else:
        # Use average yearly change from the 12th year to the end
        start_value = additionality_data['additionality_mean'].iloc[start_calc_year_index]
        end_value = additionality_data['additionality_mean'].iloc[-1]
        num_years = len(additionality_data) - 1 - start_calc_year_index # Number of intervals
        if num_years <= 0:
             logging.warning("Cannot calculate rate between start and end year (num_years <= 0). Returning 0.")
             return 0.0
        # Use index value for logging year if index is year
        log_year = additionality_data.index[start_calc_year_index] if isinstance(additionality_data.index, pd.RangeIndex) else start_calc_year_index
        rate = (end_value - start_value) / num_years
        logging.debug(f"Calculated long-term rate (from year/index {log_year}): {rate}")

    return rate

# --- forecast_additionality function (remains the same) ---
def forecast_additionality(
    additionality_data: pd.DataFrame,
    project_end_year: int, # Year sequestration stops
    rate: float
) -> pd.DataFrame:
    """
    Forecasts additionality values until the project end year, then simulates release
    until additionality reaches zero.

    Args:
        additionality_data: DataFrame with historical data, including 'year' and 'additionality_mean'.
        project_end_year: The last year the project actively sequesters carbon.
        rate: The calculated yearly rate of change for additionality.

    Returns:
        A DataFrame containing the forecasted years and additionality values.
    """
    if additionality_data.empty:
        logging.error("Input additionality_data is empty. Cannot forecast.")
        return pd.DataFrame(columns=['year', 'additionality']) # Return empty DataFrame

    # --- Initialization ---
    # Ensure 'year' column exists before accessing iloc
    if 'year' not in additionality_data.columns or 'additionality_mean' not in additionality_data.columns:
         logging.error("Missing 'year' or 'additionality_mean' column in forecast input.")
         return pd.DataFrame(columns=['year', 'additionality'])

    last_hist_year = additionality_data['year'].iloc[-1]
    last_hist_additionality = additionality_data['additionality_mean'].iloc[-1]

    forecast_data = [] # List to store forecast rows (as dicts)

    # Add the last historical point as the starting point for the forecast
    if last_hist_additionality <= 0:
         logging.warning(f"Last historical additionality ({last_hist_additionality} in year {last_hist_year}) is not positive. Forecast will only contain this point.")
         forecast_data.append({'year': last_hist_year, 'additionality': max(0, last_hist_additionality)}) # Store 0 if negative
         return pd.DataFrame(forecast_data)

    forecast_data.append({'year': last_hist_year, 'additionality': last_hist_additionality})

    current_year = last_hist_year
    current_additionality = last_hist_additionality

    # --- Sequestration Phase (if rate > 0) ---
    if rate > 0:
        while current_year < project_end_year:
            current_year += 1
            current_additionality += rate
            current_additionality = max(0, current_additionality)
            forecast_data.append({'year': current_year, 'additionality': current_additionality})
            if current_additionality <= 0:
                 logging.warning(f"Additionality became non-positive ({current_additionality}) during sequestration phase in year {current_year}. Stopping growth.")
                 break

    # --- Release Phase (or continued decline if rate <= 0) ---
    decline_rate = abs(rate) if rate != 0 else 0

    while current_additionality > 0:
        current_year += 1
        if decline_rate > 0:
             current_additionality -= decline_rate
        elif rate < 0: # Original rate was negative
             current_additionality += rate
        else: # Original rate was zero
             logging.warning("Rate is zero, additionality will not decline. Stopping forecast.")
             break

        current_additionality = max(0, current_additionality)
        forecast_data.append({'year': current_year, 'additionality': current_additionality})

        if current_year > last_hist_year + 5000:
             logging.warning("Forecast exceeded 5000 years. Stopping release phase.")
             break

    forecasted_df = pd.DataFrame(forecast_data)
    return forecasted_df


# --- damage function (updated) ---
def damage(
    forecasted_data: pd.DataFrame,
    scc_data: pd.DataFrame) -> float:
    """
    Calculates the damage modifier based on the forecasted additionality release
    profile and Social Cost of Carbon (SCC) data.

    The modifier represents the fraction of potential instantaneous damage
    (if all stock was released in the evaluation year) that is *avoided*
    due to the gradual, discounted release over time. Damage calculation
    only starts once the forecasted additionality drops below the initial level.

    Args:
        forecasted_data: DataFrame with forecasted 'year' and 'additionality'.
                         Assumes the first row represents the evaluation year state.
        scc_data: DataFrame with 'year' and 'scc' values.

    Returns:
        The damage modifier (float between 0 and potentially >1).
        Returns 1.0 if initial additionality is non-positive (no potential damage).
        Returns 0.0 if potential_damage is positive but total_damage equals or exceeds it.
    """
    if forecasted_data.empty:
        logging.error("Forecasted data is empty. Cannot calculate damage modifier.")
        return 0.0 # Or raise an error

    # --- 1. Get Initial State & Potential Damage ---
    first_row = forecasted_data.iloc[0]
    evaluation_year = int(first_row['year'])
    initial_additionality = first_row['additionality'] # The level we need to return to

    if initial_additionality <= 0:
        logging.warning(f"Initial additionality in year {evaluation_year} is non-positive ({initial_additionality}). No potential damage, modifier is 1.0.")
        return 1.0

    try:
        if 'year' not in scc_data.columns or 'scc' not in scc_data.columns:
             logging.error("Missing 'year' or 'scc' column in SCC data.")
             return 0.0
        scc_evaluation_year = scc_data.loc[scc_data['year'] == evaluation_year, 'scc'].iloc[0]
    except IndexError:
        logging.error(f"SCC value for evaluation year {evaluation_year} not found. Cannot calculate potential damage.")
        return 0.0 # Or raise error

    potential_damage = scc_evaluation_year * initial_additionality
    logging.debug(f"Evaluation Year: {evaluation_year}, Initial Additionality: {initial_additionality:.2f}")
    logging.debug(f"SCC in {evaluation_year}: {scc_evaluation_year:.2f}, Potential Damage: {potential_damage:.2f}")

    if potential_damage == 0:
         logging.warning("Potential damage calculated as zero. Modifier is 1.0.")
         return 1.0

    # --- 2. Calculate Actual Release & Identify Damage Start Year ---
    df = forecasted_data.copy()
    df['release_amount'] = df['additionality'].diff() * -1

    # Find all years where release occurs *after* the evaluation year
    all_release_df = df[(df['year'] > evaluation_year) & (df['release_amount'] > 0)].copy()

    if all_release_df.empty:
        logging.info("No release detected in forecast after evaluation year. Total damage is 0.")
        total_damage = 0.0
        damage_start_year = None # No damage calculation starts
    else:
        # Find the first year *during the release phase* where additionality drops <= initial level
        damage_calc_start_df = all_release_df[all_release_df['additionality'] <= initial_additionality]

        if damage_calc_start_df.empty:
            logging.warning(f"Forecasted additionality never dropped back to the initial level ({initial_additionality:.2f}) during the release phase. Total damage calculated as 0.")
            total_damage = 0.0
            damage_start_year = None # Damage calculation never starts
        else:
            # Get the first year this condition is met
            damage_start_year = damage_calc_start_df['year'].min()
            logging.info(f"Additionality dropped below initial level ({initial_additionality:.2f}) in year {damage_start_year}. Starting damage calculation from this year.")

            # Filter the release dataframe to include only years from the damage_start_year onwards
            release_df = all_release_df[all_release_df['year'] >= damage_start_year].copy()

            # --- 3. Calculate Discounted Damage (only from damage_start_year) ---
            release_df = pd.merge(release_df, scc_data[['year', 'scc']], on='year', how='left')
            missing_scc_years = release_df[release_df['scc'].isna()]['year'].tolist()
            if missing_scc_years:
                 logging.warning(f"Missing SCC values for release years >= {damage_start_year}: {missing_scc_years}. Damage for these years will be 0.")
                 release_df['scc'] = release_df['scc'].fillna(0.0)

            release_df['discount_factor'] = (1 + 0.03) ** (release_df['year'] - evaluation_year)
            release_df['discounted_scc_damage'] = (release_df['release_amount'] * release_df['scc']) / release_df['discount_factor']
            total_damage = release_df['discounted_scc_damage'].sum()

    logging.debug(f"Total Discounted Damage (calculated from year {damage_start_year or 'N/A'}): {total_damage:.2f}")

    # --- 4. Calculate Modifier ---
    # Potential damage remains based on the initial stock in the evaluation year
    modifier = max(0.0, (potential_damage - total_damage) / potential_damage)
    logging.info(f"Calculated Damage Modifier: {modifier:.4f}")
    return modifier

# --- Main function with argparse ---
def main():
    parser = argparse.ArgumentParser(
        description="Calculates permanence modifier based on additionality forecast and SCC, saves summary to CSV." # Updated description
    )
    parser.add_argument(
        "--additionality",
        type=str,
        required=True,
        help="Path to the input additionality CSV file (must contain 'year' and 'additionality_mean').",
    )
    parser.add_argument(
        "--scc",
        type=str,
        required=True,
        help="Path to the input SCC CSV file (must contain 'year' and 'scc').",
    )
    parser.add_argument(
        "--current_year", # This is the evaluation year for the damage calculation
        type=int,
        required=True,
        help="The evaluation year (last year of historical data used for forecast).",
    )
    parser.add_argument(
        "--project_end_year",
        type=int,
        default=2042, # Default based on previous code
        required=False,
        help="Year project stops actively sequestering carbon (used in forecast).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The destination output CSV path for permanence results summary.", # Updated help text
    )
    args = parser.parse_args()

    try:
        # Load data
        additionality_df = pd.read_csv(args.additionality)
        scc_df = pd.read_csv(args.scc)

        # Basic validation
        if 'year' not in additionality_df.columns or 'additionality_mean' not in additionality_df.columns:
             raise ValueError("Additionality CSV must contain 'year' and 'additionality_mean' columns.")
        if 'year' not in scc_df.columns or 'scc' not in scc_df.columns:
             raise ValueError("SCC CSV must contain 'year' and 'scc' columns.")

        # Filter additionality data up to the current (evaluation) year for rate calculation
        hist_additionality_df = additionality_df[additionality_df['year'] <= args.current_year].copy()
        if hist_additionality_df.empty:
             raise ValueError(f"No historical additionality data found up to current_year {args.current_year}.")

    except FileNotFoundError as e:
        logging.error(f"Input CSV not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        sys.exit(1)

    # Calculate the rate of change based on historical data
    rate = additionality_rate(hist_additionality_df)
    logging.info(f"Rate of change calculated from data up to {args.current_year}: {rate}")

    # Forecast the additionality using historical data up to current_year
    forecasted_additionality = forecast_additionality(
        hist_additionality_df, # Use only historical data to start forecast
        args.project_end_year,
        rate
    )

    logging.info(f"Forecast generated based on data up to {args.current_year}.")
    if not forecasted_additionality.empty:
        logging.info(f"Forecast ends in year: {forecasted_additionality['year'].iloc[-1]}")

        # Calculate the damage modifier using the forecast and SCC data
        damage_modifier = damage(forecasted_additionality, scc_df)

        # --- Prepare and Save Output as CSV ---
        output_data = {
            "evaluation_year": args.current_year,
            "rate_of_change": rate,
            "project_end_year": args.project_end_year,
            "forecast_end_year": forecasted_additionality['year'].iloc[-1],
            "damage_modifier": damage_modifier
        }

        # Convert dictionary to DataFrame (single row)
        output_df = pd.DataFrame([output_data])

        try:
            # Save the DataFrame to CSV
            output_df.to_csv(args.output, index=False) # Use args.output directly
            logging.info(f"Permanence results summary saved to {args.output}")
        except Exception as e:
             logging.error(f"Failed to write output CSV: {e}")
             sys.exit(1)

    else:
        logging.error("Forecast DataFrame is empty. Cannot calculate damage modifier or save results.")
        error_data = {
            "evaluation_year": args.current_year,
            "error": "Failed to generate forecast."
        }
        error_df = pd.DataFrame([error_data])
        try:
            error_df.to_csv(args.output, index=False)
            logging.info(f"Error summary saved to {args.output}")
        except Exception as e:
             logging.error(f"Failed to write error CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()