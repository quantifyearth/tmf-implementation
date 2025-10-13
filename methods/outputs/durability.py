import argparse
import pandas as pd
import numpy as np
import glob
import os
import json

def calculate_durability_metrics(
    additionality_csv_path: str,
    grid_folder_path: str,
    scc_csv_path: str,
    project_end_date: int,
    skip_rows: int = 10,
    additionality_percentile: float = 0.05,
    control_percentile: float = 0.05,
    discount_rate: float = 0.03
) -> dict:
    """
    Calculate durability metrics for a carbon project.
    """
    
    # 1. process cumulative additionality data
    Ati_df = pd.read_csv(additionality_csv_path)
    t_start = Ati_df.iloc[skip_rows]['year'].astype(int)
    t_now = Ati_df.iloc[-1]['year'].astype(int)
    
    # calculate differences
    ati_df = Ati_df[(Ati_df['year'] >= t_start) & (Ati_df['year'] <= t_now)]
    ati_df = ati_df[['year', 'additionality_mean']].copy()
    ati_df['ati'] = ati_df['additionality_mean'].diff().fillna(0)
    ati_df = ati_df[['year', 'ati']].copy()
    
    # 2. grid level additionality data for aomega
    csv_files = glob.glob(os.path.join(grid_folder_path, "*.csv"))
    Gnati = []
    
    for f in csv_files:
        grid = pd.read_csv(f)
        grid = grid.iloc[skip_rows:-1]
        grid['ati'] = grid['additionality'].diff()
        grid = grid.dropna()
        ati = grid['ati'].tolist()
        Gnati.extend(ati)
    
    aomega = np.percentile(Gnati, additionality_percentile * 100)
    
    # if aomega is positive, set to zero
    if aomega > 0:
        aomega = 0
    
    
    # 3. grid level control carbon data for somega
    CGnatis = []
    
    for f in csv_files:
        grid = pd.read_csv(f)
        grid = grid.iloc[skip_rows:-1]
        grid['ati'] = grid['control_carbon'].diff()
        grid = grid.dropna()
        ati = grid['ati'].tolist()
        CGnatis.extend(ati)
    
    somega = np.percentile(CGnatis, control_percentile * 100)
    
    # 4. time series with future projections
    future_years1 = pd.DataFrame({
        'year': range(t_now + 1, project_end_date), 
        'ati': aomega
    })
    future_years2 = pd.DataFrame({
        'year': range(project_end_date + 1, project_end_date + 100), 
        'ati': somega
    })
    ati_df = pd.concat([ati_df, future_years1, future_years2], ignore_index=True)
    
    # 5. fill negative values with positive values
    fill_records = []
    ati_df_filled = ati_df.copy()
    
    while True:
        neg_idx = ati_df_filled[ati_df_filled['ati'] < 0].index
        if len(neg_idx) == 0:
            break
        
        first_neg_idx = neg_idx[0]
        pos_idx = ati_df_filled.loc[:first_neg_idx-1][ati_df_filled['ati'] > 0].index
        
        if len(pos_idx) == 0:
            ati_df_filled.loc[first_neg_idx, 'ati'] = 0
            continue
        
        last_pos_idx = pos_idx[-1]
        fill_value = ati_df_filled.loc[last_pos_idx, 'ati']
        neg_value = -ati_df_filled.loc[first_neg_idx, 'ati']
        amount_filled = min(neg_value, fill_value)
        
        if amount_filled > 0:
            fill_records.append({
                'year_of_source': ati_df_filled.loc[last_pos_idx, 'year'],
                'year_of_filling': ati_df_filled.loc[first_neg_idx, 'year'],
                'amount_filled': amount_filled
            })
            ati_df_filled.loc[last_pos_idx, 'ati'] -= amount_filled
            ati_df_filled.loc[first_neg_idx, 'ati'] += amount_filled
    
    fill_df = pd.DataFrame(fill_records)

    # 6. ratios and metrics
    if len(fill_df) > 0:
        scc_df = pd.read_csv(scc_csv_path)
        scc_df['discount'] = scc_df['scc'] / ((1 + discount_rate) ** (scc_df['year'] - t_now))
        
        fill_df['scc_ratio'] = 1 - (
            fill_df['year_of_filling'].map(scc_df.set_index('year')['discount']) / 
            fill_df['year_of_source'].map(scc_df.set_index('year')['discount'])
        )
        
        # make release df just up until current year
        release_df = fill_df[fill_df['year_of_source'] <= t_now]
        
        if len(release_df) > 0:
            weighted_mean_ep = (release_df['amount_filled'] * release_df['scc_ratio']).sum() / release_df['amount_filled'].sum()
            weighted_mean_lifespan = (release_df['amount_filled'] * (release_df['year_of_filling'] - release_df['year_of_source'])).sum() / release_df['amount_filled'].sum()

            # mean eP and lifespan per year of source (only for historical period)
            mean_ep_per_year = (release_df['amount_filled'] * release_df['scc_ratio']).groupby(release_df['year_of_source']).sum() / release_df['amount_filled'].groupby(release_df['year_of_source']).sum()
            mean_lifespan_per_year = (release_df['amount_filled'] * (release_df['year_of_filling'] - release_df['year_of_source'])).groupby(release_df['year_of_source']).sum() / release_df['amount_filled'].groupby(release_df['year_of_source']).sum()
        else:
            weighted_mean_ep = 0
            weighted_mean_lifespan = 0
            mean_ep_per_year = pd.Series()
            mean_lifespan_per_year = pd.Series()
    else:
        weighted_mean_ep = 0
        weighted_mean_lifespan = 0
        mean_ep_per_year = pd.Series()
        mean_lifespan_per_year = pd.Series()
    
    return {
        'weighted_mean_ep': weighted_mean_ep,
        'weighted_mean_lifespan': weighted_mean_lifespan,
        'fill_df': fill_df,
        'release_df': release_df,
        'ati_df_filled': ati_df_filled,
        't_start': t_start,
        't_now': t_now,
        'aomega': aomega,
        'somega': somega,
        'mean_ep_per_year': mean_ep_per_year,
        'mean_lifespan_per_year': mean_lifespan_per_year
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate durability metrics for carbon projects")
    
    # arguments
    parser.add_argument("--additionality-csv", required=True,
                       help="Path to CSV file with yearly additionality_mean values")
    parser.add_argument("--grid-folder", required=True,
                       help="Path to folder containing grid CSV files with additionality and control_carbon")
    parser.add_argument("--scc-csv", required=True,
                       help="Path to CSV file with social cost of carbon data")
    parser.add_argument("--project-end-date", type=int, required=True,
                       help="Year when project ends (e.g., 2041)")
    parser.add_argument("--output-dir", required=True,
                       help="Directory where 'durability' folder will be created")
    
    # optional arguments
    parser.add_argument("--skip-rows", type=int, default=10,
                       help="Number of initial rows to skip in processing (default: 10)")
    parser.add_argument("--additionality-percentile", type=float, default=0.05,
                       help="Percentile for additionality omega calculation (default: 0.05)")
    parser.add_argument("--control-percentile", type=float, default=0.05,
                       help="Percentile for control omega calculation (default: 0.05)")
    parser.add_argument("--discount-rate", type=float, default=0.03,
                       help="Discount rate for SCC calculations (default: 0.03)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    args = parser.parse_args()
    
    # durability output folder
    durability_dir = os.path.join(args.output_dir, "durability")
    os.makedirs(durability_dir, exist_ok=True)
    
    # durability metrics
    results = calculate_durability_metrics(
        additionality_csv_path=args.additionality_csv,
        grid_folder_path=args.grid_folder,
        scc_csv_path=args.scc_csv,
        project_end_date=args.project_end_date,
        skip_rows=args.skip_rows,
        additionality_percentile=args.additionality_percentile,
        control_percentile=args.control_percentile,
        discount_rate=args.discount_rate
    )
    
    # outputs
    # 1. fill_df as CSV
    if len(results['fill_df']) > 0:
        fill_df_path = os.path.join(durability_dir, "fill_records.csv")
        results['fill_df'].to_csv(fill_df_path, index=False)
        print(f"Fill records saved to: {fill_df_path}")
    
    # 2. mean eP per year as CSV
    if len(results['mean_ep_per_year']) > 0:
        mean_ep_df = pd.DataFrame({
            'year_of_source': results['mean_ep_per_year'].index,
            'mean_ep': results['mean_ep_per_year'].values
        })
        mean_ep_path = os.path.join(durability_dir, "mean_ep_per_year.csv")
        mean_ep_df.to_csv(mean_ep_path, index=False)
        print(f"Mean eP per year saved to: {mean_ep_path}")
    
    # 3. mean lifespan per year as CSV
    if len(results['mean_lifespan_per_year']) > 0:
        mean_lifespan_df = pd.DataFrame({
            'year_of_source': results['mean_lifespan_per_year'].index,
            'mean_lifespan': results['mean_lifespan_per_year'].values
        })
        mean_lifespan_path = os.path.join(durability_dir, "mean_lifespan_per_year.csv")
        mean_lifespan_df.to_csv(mean_lifespan_path, index=False)
        print(f"Mean lifespan per year saved to: {mean_lifespan_path}")
    
    # 4. weighted means and key parameters as JSON
    summary_data = {
        "weighted_mean_ep": float(results['weighted_mean_ep']),
        "weighted_mean_lifespan": float(results['weighted_mean_lifespan']),
        "t_start": int(results['t_start']),
        "t_now": int(results['t_now']),
        "project_end_date": int(args.project_end_date),
        "aomega": float(results['aomega']),
        "somega": float(results['somega']),
        "parameters": {
            "skip_rows": args.skip_rows,
            "additionality_percentile": args.additionality_percentile,
            "control_percentile": args.control_percentile,
            "discount_rate": args.discount_rate
        },
        "input_files": {
            "additionality_csv": args.additionality_csv,
            "grid_folder": args.grid_folder,
            "scc_csv": args.scc_csv
        }
    }
    
    summary_path = os.path.join(durability_dir, "durability_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # results
    print(f"\n=== DURABILITY METRICS ===")
    print(f"Weighted Mean eP: {results['weighted_mean_ep']:.6f}")
    print(f"Weighted Mean Lifespan: {results['weighted_mean_lifespan']:.2f}")
    print(f"Start Year: {results['t_start']}")
    print(f"Current Year: {results['t_now']}")
    print(f"Additionality Omega: {results['aomega']:.6f}")
    print(f"Control Omega: {results['somega']:.6f}")
    
    if args.verbose:
        if len(results['release_df']) > 0:
            print(f"Total amount filled: {results['release_df']['amount_filled'].sum():.6f}")
            print(f"Average fill amount: {results['release_df']['amount_filled'].mean():.6f}")
    
    print(f"\nAll outputs saved to: {durability_dir}")

if __name__ == "__main__":
    main()