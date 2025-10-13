#!/usr/bin/env python3

# ----------------------------
# import necessary libraries
# ----------------------------
import os
import argparse
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm
from typing import Set, List
import logging
import numpy as np

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INITIAL_CODES = {1}  # natural land use codes
ARABLE_CODES = {2,3,4,5,6}         # converted land use codes


def clip_raster_cells_to_project(life_path: str, project_path: str) -> gpd.GeoDataFrame:
    """
    raster cells to the project area boundary and extracts band values.

    parameters:
    - life_path: input raster file
    - project_path: input shapefile or geojson

    returns:
    - gdf with clipped grid and band values
    """
    project = gpd.read_file(project_path)
    logging.info(f"Loaded project area from {project_path}")

    with rasterio.open(life_path) as src:
        project = project.to_crs(src.crs)
        logging.info(f"Reprojected project area to CRS {src.crs}")

        minx, miny, maxx, maxy = project.total_bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)

        # get band names
        band_names = src.descriptions
        if not band_names or any(name is None for name in band_names):
            band_names = [f"band_{i+1}" for i in range(out_image.shape[0])]
            logging.warning("Band names not found in raster metadata. Assigned default band names.")
        else:
            band_names = [name.lower() if name else f"band_{i+1}" for i, name in enumerate(band_names)]
            logging.info(f"Extracted band names: {band_names}")

    # initialise data structure
    data = {'geometry': []}
    for band in band_names:
        data[band] = []

    height, width = out_image.shape[1], out_image.shape[2]
    x_min, y_max = out_transform[2], out_transform[5]
    pixel_size_x, pixel_size_y = out_transform[0], -out_transform[4]

    project_union = unary_union(project.geometry)

    # process each pixel
    for row in tqdm(range(height), desc="Processing rows"):
        for col in range(width):
            x_left = x_min + col * pixel_size_x
            x_right = x_left + pixel_size_x
            y_top = y_max - row * pixel_size_y
            y_bottom = y_top - pixel_size_y

            pixel_polygon = box(x_left, y_bottom, x_right, y_top)

            if project_union.intersects(pixel_polygon):
                intersection = project_union.intersection(pixel_polygon)

                if intersection.is_empty or not intersection.geom_type in ['Polygon', 'MultiPolygon']:
                    continue

                if intersection.geom_type == 'MultiPolygon':
                    for geom in intersection.geoms:
                        data['geometry'].append(geom)
                        for i, band in enumerate(band_names):
                            data[band].append(out_image[i, row, col])
                else:
                    data['geometry'].append(intersection)
                    for i, band in enumerate(band_names):
                        data[band].append(out_image[i, row, col])

    grid_gdf = gpd.GeoDataFrame(data, crs=src.crs)
    logging.info(f"Clipped raster cells to project area. Total grid cells: {len(grid_gdf)}")
    return grid_gdf


def flag_additionality_vectorized(luc_start: pd.Series, luc_eval: pd.Series) -> pd.Series:
    """
    flags change from natural to converted land use codes.

    parameters:
    - luc_start: Land use codes at start year
    - luc_eval: Land use codes at evaluation year

    returns:
    - binary series where 1 represents natural→converted, else 0
    """
    mask_initial = luc_start.isin(INITIAL_CODES)
    mask_final = luc_eval.isin(ARABLE_CODES)
    return (mask_initial & mask_final).astype(int)


def process_paired_data(parquet_folder: str, eval_years: List[int], min_year: int, max_year: int) -> dict:
    """
    processes paired parquet files to determine land use changes.

    parameters:
    - parquet_folder: parquet files
    - eval_years: evaluation years
    - min_year: start year
    - max_year: end year

    returns:
    - each evaluation year with corresponding GeoDataFrame
    """
    # load paired parquet files
    all_files = [
        os.path.join(parquet_folder, f) for f in os.listdir(parquet_folder)
        if f.endswith('.parquet') and 'matchless' not in f.lower()
    ]

    if not all_files:
        raise FileNotFoundError(f"No paired parquet files found in '{parquet_folder}'")

    logging.info(f"Found {len(all_files)} paired parquet files to process")

    # combine all paired data
    data_frames = []
    for file in all_files:
        logging.info(f"Reading paired file: {file}")
        try:
            df = pd.read_parquet(file)
            data_frames.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue

    if not data_frames:
        raise ValueError("No data loaded from paired parquet files")

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df.reset_index(drop=True).reset_index().rename(columns={'index': 'p_id'})
    logging.info(f"Combined paired dataframe shape: {combined_df.shape}")

    # validate required columns
    required_columns = ['k_lng', 'k_lat', 's_lng', 's_lat']
    for year in eval_years:
        required_columns.extend([
            f'k_luc_{min_year}', f'k_luc_{year}',
            f's_luc_{min_year}', f's_luc_{year}'
        ])

    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # create gdfs for project and counterfactual points
    project_gdf = gpd.GeoDataFrame(
        combined_df[['p_id', 'k_lng', 'k_lat']].copy(),
        geometry=gpd.points_from_xy(combined_df['k_lng'], combined_df['k_lat']),
        crs="EPSG:4326"
    )

    cf_gdf = gpd.GeoDataFrame(
        combined_df[['p_id', 's_lng', 's_lat']].copy(),
        geometry=gpd.points_from_xy(combined_df['s_lng'], combined_df['s_lat']),
        crs="EPSG:4326"
    )

    year_results = {}

    # process each evaluation year
    for year in eval_years:
        logging.info(f"Processing year: {year}")

        # prepare project data
        project_gdf_year = project_gdf.copy()
        project_gdf_year['luc_start'] = combined_df[f'k_luc_{min_year}']
        project_gdf_year['luc_eval'] = combined_df[f'k_luc_{year}']
        project_gdf_year['project_additionality'] = flag_additionality_vectorized(
            project_gdf_year['luc_start'], project_gdf_year['luc_eval']
        )

        # prepare counterfactual data
        cf_gdf_year = cf_gdf.copy()
        cf_gdf_year['luc_start'] = combined_df[f's_luc_{min_year}']
        cf_gdf_year['luc_eval'] = combined_df[f's_luc_{year}']
        cf_gdf_year['counter_factual_additionality'] = flag_additionality_vectorized(
            cf_gdf_year['luc_start'], cf_gdf_year['luc_eval']
        )

        # merge project and counterfactual results
        final_gdf = project_gdf_year[['p_id', 'geometry', 'project_additionality']].merge(
            cf_gdf_year[['p_id', 'counter_factual_additionality']],
            on='p_id',
            how='left'
        )

        final_gdf['difference_additionality'] = (
            final_gdf['counter_factual_additionality'] - final_gdf['project_additionality']
        )

        year_results[year] = final_gdf

    return year_results


def process_matchless_data(parquet_folder: str, grid_gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Processes matchless parquet files to count matchless points per grid cell.

    Parameters:
    - parquet_folder: Path to parquet files
    - grid_gdf: GeoDataFrame of grid cells with 'grid_id'

    Returns:
    - Series mapping 'grid_id' to matchless point counts
    """
    matchless_files = [
        os.path.join(parquet_folder, f) for f in os.listdir(parquet_folder)
        if f.endswith('.parquet') and 'matchless' in f.lower()
    ]

    if not matchless_files:
        logging.warning(f"No matchless parquet files found in '{parquet_folder}'")
        return pd.Series(dtype=int)

    logging.info(f"Found {len(matchless_files)} matchless parquet files to process")

    # load and combine matchless data
    matchless_dfs = []
    for file in matchless_files:
        try:
            df = pd.read_parquet(file)
            matchless_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue

    if not matchless_dfs:
        logging.warning("No data loaded from matchless parquet files")
        return pd.Series(dtype=int)

    combined_matchless_df = pd.concat(matchless_dfs, ignore_index=True)
    logging.info(f"Combined matchless dataframe shape: {combined_matchless_df.shape}")

    if not {'lng', 'lat'}.issubset(combined_matchless_df.columns):
        raise ValueError("Matchless parquet files must contain 'lng' and 'lat' columns")

    # create gdf for matchless points
    matchless_gdf = gpd.GeoDataFrame(
        combined_matchless_df,
        geometry=gpd.points_from_xy(combined_matchless_df['lng'], combined_matchless_df['lat']),
        crs="EPSG:4326"
    )

    # ensure CRS compatibility
    if grid_gdf.crs != matchless_gdf.crs:
        grid_gdf = grid_gdf.to_crs(matchless_gdf.crs)

    # spatial join to assign matchless points to grid cells
    matchless_with_grid = gpd.sjoin(matchless_gdf, grid_gdf[['grid_id', 'geometry']], how='left', predicate='within')
    matchless_with_grid = matchless_with_grid.dropna(subset=['grid_id'])
    
    logging.info(f"Matchless points within grid: {len(matchless_with_grid)}")
    
    return matchless_with_grid.groupby('grid_id').size()


def compute_biodiversity_summary(
    year_results: dict,
    grid_geojson_path: str,
    output_folder: str,
    output_csv: str,
    band_columns: List[str],
    matchless_counts: pd.Series,
    min_points: int = 20
) -> None:
    """
    biodiversity additionality summarise - export results.
    """
    grid_gdf_original = gpd.read_file(grid_geojson_path)
    logging.info(f"Loaded clipped raster geojson from {grid_geojson_path}")

    # ensure grid_id exists
    if 'grid_id' not in grid_gdf_original.columns:
        grid_gdf_original = grid_gdf_original.reset_index().rename(columns={'index': 'grid_id'})

    # get ALL band columns from the LIFE grid
    available_bands = [col for col in grid_gdf_original.columns 
                      if col not in ['grid_id', 'geometry']]
    
    logging.info(f"Processing ALL available band columns: {available_bands}")

    # data structure
    csv_data = {'year': []}
    for band in available_bands:
        csv_data[f'{band}_add'] = []

    os.makedirs(output_folder, exist_ok=True)

    for year, final_gdf in year_results.items():
        logging.info(f"Processing year: {year}")

        # join points with grid cells
        points_within_grid = gpd.sjoin(final_gdf, grid_gdf_original, how='left', predicate='within')
        points_within_grid = points_within_grid.dropna(subset=['grid_id'])
        logging.info(f"Paired points within grid: {len(points_within_grid)}")

        # aggregate by grid cell
        aggregation = points_within_grid.groupby('grid_id').agg(
            natural_land_project_flags=('project_additionality', 'sum'),
            natural_land_counterfactual_flags=('counter_factual_additionality', 'sum'),
            paired_points=('p_id', 'count')
        ).reset_index()

        # merge with matchless counts
        if not matchless_counts.empty:
            aggregation = aggregation.merge(
                matchless_counts.rename('matchless_points').reset_index(),
                on='grid_id',
                how='left'
            )
            aggregation['matchless_points'] = aggregation['matchless_points'].fillna(0).astype(int)
        else:
            aggregation['matchless_points'] = 0

        aggregation['total_points'] = aggregation['paired_points'] + aggregation['matchless_points']

        # filter by minimum points threshold
        aggregation = aggregation[aggregation['total_points'] >= min_points]
        logging.info(f"Grid cells after filtering with min_points ({min_points}): {len(aggregation)}")

        # calculate conversion proportions
        aggregation['proportion_project'] = (
            aggregation['natural_land_project_flags'] / aggregation['total_points']
        ).clip(upper=1)
        aggregation['proportion_counterfactual'] = (
            aggregation['natural_land_counterfactual_flags'] / aggregation['total_points']
        ).clip(upper=1)

        # merge with grid geometry
        grid_with_aggregation = grid_gdf_original.merge(
            aggregation[['grid_id', 'proportion_project', 'proportion_counterfactual', 
                        'total_points', 'paired_points', 'matchless_points']],
            on='grid_id',
            how='left'
        )

        grid_with_aggregation[['proportion_project', 'proportion_counterfactual']] = (
            grid_with_aggregation[['proportion_project', 'proportion_counterfactual']].fillna(0)
        )

        # calculate areas
        grid_with_area = grid_with_aggregation.to_crs('ESRI:54034')
        grid_with_aggregation['cell_area_km2'] = grid_with_area['geometry'].area / 1e6

        grid_with_aggregation['natural_land_project_km2'] = (
            grid_with_aggregation['proportion_project'] * grid_with_aggregation['cell_area_km2']
        )
        grid_with_aggregation['natural_land_counterfactual_km2'] = (
            grid_with_aggregation['proportion_counterfactual'] * grid_with_aggregation['cell_area_km2']
        )
        grid_with_aggregation['difference_km2'] = (
            grid_with_aggregation['natural_land_counterfactual_km2'] - 
            grid_with_aggregation['natural_land_project_km2']
        )

        # calculate biodiversity additionality for bands 
        for band in available_bands:
            if band not in grid_with_aggregation.columns:
                raise KeyError(f"Band column '{band}' is missing in the grid GeoDataFrame")

            add_col = f"{band}_add"
            grid_with_aggregation[add_col] = (
                grid_with_aggregation['difference_km2'] * grid_with_aggregation[band]
            )

        # store data
        csv_data['year'].append(year)
        for band in available_bands:
            add_col = f'{band}_add'
            csv_data[add_col].append(grid_with_aggregation[add_col].sum())

        # yearly geojson with bands
        geojson_fields = {
            'year': year,
            'area_km2': grid_with_aggregation['cell_area_km2'],
            'natural_land_project_km2': grid_with_aggregation['natural_land_project_km2'],
            'natural_land_counterfactual_km2': grid_with_aggregation['natural_land_counterfactual_km2'],
            'natural_land_difference_km2': grid_with_aggregation['difference_km2'],
            'total_points': grid_with_aggregation.get('total_points', 0),
            'paired_points': grid_with_aggregation.get('paired_points', 0),
            'matchless_points': grid_with_aggregation.get('matchless_points', 0),
        }

        # band values and additionality
        for band in available_bands:
            geojson_fields[band] = grid_with_aggregation[band]
            geojson_fields[f'{band}_additionality'] = grid_with_aggregation[f'{band}_add']

        geojson_gdf = gpd.GeoDataFrame(
            geojson_fields,
            geometry=grid_with_aggregation['geometry'],
            crs=grid_gdf_original.crs
        )

        output_geojson = os.path.join(output_folder, f"biodiversity_additionality_{year}.geojson")
        geojson_gdf.to_file(output_geojson, driver="GeoJSON")
        logging.info(f"Biodiversity additionality map for year {year} saved to: {output_geojson}")

    # summary with bands
    cols = ['year'] + [f'{band}_add' for band in available_bands]
    summary_df = pd.DataFrame(csv_data, columns=cols)

    summary_df.to_csv(output_csv, index=False)
    logging.info(f"Biodiversity summary CSV saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Process biodiversity raster and paired matching data for additionality analysis"
    )

    parser.add_argument("--life", required=True, help="Path to biodiversity raster (GeoTIFF)")
    parser.add_argument("--project", required=True, help="Path to project area (GeoJSON)")
    parser.add_argument("--output_clip", required=True, help="Output path for clipped biodiversity grid (GeoJSON)")
    parser.add_argument("--parquet_folder", required=True, help="Directory containing paired and matchless parquet files")
    parser.add_argument("--output_folder", required=True, help="Output directory for yearly biodiversity maps")
    parser.add_argument("--output_csv", required=True, help="Output path for biodiversity summary CSV")
    parser.add_argument("--min_points", type=int, default=20, help="Minimum points per grid cell (default: 20)")

    args = parser.parse_args()

    # input files
    for path, name in [(args.life, "biodiversity raster"), (args.project, "project area")]:
        if not os.path.exists(path):
            logging.error(f"File not found: {path} ({name})")
            exit(1)

    if not os.path.isdir(args.parquet_folder):
        logging.error(f"Directory not found: {args.parquet_folder}")
        exit(1)

    try:
        # biodiversity raster to project area
        logging.info("Clipping biodiversity raster to project area...")
        clipped_grid = clip_raster_cells_to_project(args.life, args.project)

        if 'grid_id' not in clipped_grid.columns:
            clipped_grid = clipped_grid.reset_index().rename(columns={'index': 'grid_id'})

        clipped_grid.to_file(args.output_clip, driver="GeoJSON")
        logging.info(f"Clipped biodiversity grid saved to {args.output_clip}")

        # infer year range from parquet files
        logging.info("Inferring year range from parquet files...")
        paired_files = [
            os.path.join(args.parquet_folder, f) for f in os.listdir(args.parquet_folder)
            if f.endswith('.parquet') and 'matchless' not in f.lower()
        ]

        if not paired_files:
            raise FileNotFoundError(f"No paired parquet files found in {args.parquet_folder}")

        all_years = set()
        for file in paired_files:
            df = pd.read_parquet(file, columns=None, engine='pyarrow')
            luc_columns = [col for col in df.columns if col.startswith(('k_luc_', 's_luc_'))]
            years = [int(col.split('_')[-1]) for col in luc_columns if col.split('_')[-1].isdigit()]
            all_years.update(years)

        if not all_years:
            raise ValueError("No valid years found in parquet files")

        min_year, max_year = min(all_years), max(all_years)
        start_year = min_year + 10
        eval_years = list(range(start_year + 1, max_year + 1))

        logging.info(f"Start year: {start_year}, Evaluation years: {eval_years}")

        # paired data
        logging.info("Processing paired matching data...")
        year_results = process_paired_data(args.parquet_folder, eval_years, start_year, max_year)

        # matchless data
        logging.info("Processing matchless data...")
        matchless_counts = process_matchless_data(args.parquet_folder, clipped_grid)

        # biodiversity summaries
        logging.info("Computing biodiversity additionality summaries...")
        compute_biodiversity_summary(
            year_results=year_results,
            grid_geojson_path=args.output_clip,
            output_folder=args.output_folder,
            output_csv=args.output_csv,
            band_columns=[],  # function will discover all bands
            matchless_counts=matchless_counts,
            min_points=args.min_points
        )

        logging.info("Biodiversity additionality analysis completed successfully!")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        exit(1)


if __name__ == "__main__":
    main()
