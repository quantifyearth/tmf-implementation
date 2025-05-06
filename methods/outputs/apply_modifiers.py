import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

def calculate_and_report_total(
    additionality_file: str,
    permanence_file: str, # Now expects CSV path
    leakage_rate: float,
    output_txt_file: str
) -> None:
    """
    Applies leakage and permanence modifiers, calculates the total adjusted
    additionality, and reports it to console and a text file.

    Args:
        additionality_file: Path to the additionality CSV file (must contain 'year' and 'additionality_mean').
        permanence_file: Path to the permanence summary CSV file (must contain 'damage_modifier').
        leakage_rate: The fixed leakage rate (e.g., 0.40 for 40%).
        output_txt_file: Path to save the output summary text message.
    """
    try:
        # Load additionality data
        add_df = pd.read_csv(additionality_file)
        logging.info(f"Loaded additionality data from {additionality_file}")

        # Check for expected columns in additionality data
        if 'additionality_mean' not in add_df.columns or 'year' not in add_df.columns:
            logging.error(f"'additionality_mean' and/or 'year' column not found in {additionality_file}")
            sys.exit(1)

        # Load permanence summary data from CSV
        perm_df = pd.read_csv(permanence_file)
        logging.info(f"Loaded permanence summary data from {permanence_file}")

        # Validate permanence DataFrame and extract damage modifier
        if perm_df.empty:
            logging.error(f"Permanence summary CSV file is empty: {permanence_file}")
            sys.exit(1)
        if len(perm_df) > 1:
            logging.warning(f"Permanence summary CSV has multiple rows. Using data from the first row only.")
        if 'damage_modifier' not in perm_df.columns:
            logging.error(f"'damage_modifier' column not found in {permanence_file}")
            sys.exit(1)

        # Extract damage modifier from the first row
        damage_modifier = perm_df['damage_modifier'].iloc[0]
        logging.info(f"Using Damage Modifier: {damage_modifier:.4f}")

        # Calculate leakage multiplier
        leakage_multiplier = 1.0 - leakage_rate
        logging.info(f"Using Leakage Multiplier (1 - {leakage_rate:.2f}): {leakage_multiplier:.4f}")

        # Apply modifiers
        add_df['final_additionality'] = add_df['additionality_mean'] * leakage_multiplier * damage_modifier
        logging.info("Applied leakage rate and permanence damage modifier.")

        # Calculate the total sum
        total_avoided_emissions = add_df['final_additionality'].iloc[-1]

        # Get start and end year from the additionality data
        start_year = add_df['year'].iloc[10]
        end_year = add_df['year'].max()

        # Format the output message
        output_message = (
            f"Total emissions of carbon dioxide tonnes likely avoided by intervention "
            f"from {start_year} to {end_year} is: {total_avoided_emissions:,.2f} eCO2"
        )

        # Print to console
        print("\n" + "="*len(output_message))
        print(output_message)
        print("="*len(output_message) + "\n")

        # Write to output text file
        with open(output_txt_file, 'w') as f:
            f.write(output_message + "\n")
        logging.info(f"Summary message saved to {output_txt_file}")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found - {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"Error: Input CSV file is empty - {permanence_file}")
        sys.exit(1)
    except KeyError as e: # Could still happen if column name is wrong
        logging.error(f"Error: Missing expected column in data - {e}")
        sys.exit(1)
    except IndexError as e: # If iloc[0] fails on empty df (already checked, but good practice)
        logging.error(f"Error accessing data in permanence CSV (possibly empty after filtering?): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate total adjusted additionality and report summary."
    )
    parser.add_argument(
        "--additionality",
        type=str,
        required=True,
        help="Path to the input additionality CSV file.",
    )
    parser.add_argument(
        "--permanence",
        type=str,
        required=True,
        help="Path to the input permanence summary CSV file.", # Updated help text
    )
    parser.add_argument(
        "--leakage",
        type=float,
        required=True,
        help="Fixed leakage rate (e.g., 0.40 for 40%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output summary text file.",
    )

    args = parser.parse_args()

    calculate_and_report_total(
        args.additionality,
        args.permanence,
        args.leakage,
        args.output
    )