from typing import Dict, List
import os
import glob
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
from yirgacheffe.layers import RasterLayer, GroupLayer  # type: ignore

def generate_carbon_density(
    jrc_directory_path: str,
    gedi_data_file: str,
    output_file: str
) -> None:
    _, output_ext = os.path.splitext(output_file)
    output_ext = output_ext.lower()
    if output_ext not in ['.csv', '.parquet']:
        raise ValueError("We only support .csv and .parquet outputs.")

    # Define the below-ground conversion factors based on region and AGBD
    bgbd_factors = {
        "Africa": {
            "above_125": 0.232,
            "below_125": 0.232
        },
        "Americas": {
            "above_125": 0.284,
            "below_125": 0.2845
        },
        "Asia": {
            "above_125": 0.246,
            "below_125": 0.323
        }
    }

    # Load the JRC raster layers
    jrc_files = glob.glob("*2024*.tif", root_dir=jrc_directory_path)
    if not jrc_files:
        raise ValueError(f"No TIF files found in {jrc_directory_path}")

    # Create raster layers and track regions
    raster_layers = []
    raster_layer_info = []
    region_codes = {"AFR": "Africa", "ASI": "Asia", "SAM": "Americas"}
    
    for filename in jrc_files:
        full_path = os.path.join(jrc_directory_path, filename)
        try:
            layer = RasterLayer.layer_from_file(full_path)
            raster_layers.append(layer)
            
            region_found = False
            for code, region in region_codes.items():
                if code in filename:
                    raster_layer_info.append((layer, region, filename))
                    region_found = True
                    break
            
            if not region_found:
                raster_layer_info.append((layer, "Unknown", filename))
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not raster_layers:
        raise ValueError("No valid raster layers could be loaded")

    # Create the group layer from all loaded rasters
    jrc_raster_layer = GroupLayer(raster_layers, name="jrc_layer")
    
    # Load GEDI points
    gedi = gpd.read_file(gedi_data_file)
    
    # Track which regions actually contain GEDI points
    region_point_counts = {"Africa": 0, "Americas": 0, "Asia": 0, "Unknown": 0}
    pixel_scale = jrc_raster_layer.pixel_scale
    luc_buckets: Dict[int, List[float]] = {}

    for _, row in gedi.iterrows():
        agbd = row['agbd']
        point = row['geometry']

        try:
            # Find which specific raster layer contains this point
            matching_layer_info = None
            for layer_info in raster_layer_info:
                layer, region, _ = layer_info
                if (layer.area.left <= point.x <= layer.area.right and 
                    layer.area.top >= point.y >= layer.area.bottom):
                    # This layer contains the point
                    matching_layer_info = layer_info
                    region_point_counts[region] += 1
                    break
            
            if matching_layer_info is None:
                continue

            # Use group layer to get the LUC value
            xoffset = math.floor((point.x - jrc_raster_layer.area.left) / pixel_scale.xstep)
            yoffset = math.floor((point.y - jrc_raster_layer.area.top) / pixel_scale.ystep)
            surroundings = jrc_raster_layer.read_array(xoffset - 1, yoffset - 1, 3, 3)
            land_use_class = surroundings[1][1]

            if not np.all(surroundings == land_use_class):
                continue  # Skip points near boundaries

            try:
                luc_buckets[land_use_class].append(agbd)
            except KeyError:
                luc_buckets[land_use_class] = [agbd]
            
        except Exception:
            continue

    # Determine the region based on where most GEDI points are located
    if sum(region_point_counts.values()) == 0:
        region = "Americas"  # Default
    else:
        # Exclude "Unknown" region unless it's the only one with points
        if sum(count for r, count in region_point_counts.items() if r != "Unknown") > 0:
            region_point_counts["Unknown"] = 0
        region = max(region_point_counts.items(), key=lambda x: x[1])[0]
    
    # Process results
    results = []
    for land_use_class in sorted(luc_buckets.keys()):
        vals = np.array(luc_buckets[land_use_class], dtype=np.float64)
        vals.sort()
        n0 = vals.size
        trim = int(math.floor(n0 * 0.25))
        mid_vals = vals[trim : n0 - trim]
        n_mid = mid_vals.size

        midmean = np.mean(mid_vals) if n_mid > 0 else np.nan
        agbd_above_125 = midmean > 125
        key = "above_125" if agbd_above_125 else "below_125"
        bgbd_factor = bgbd_factors.get(region, bgbd_factors["Americas"])[key]
        
        carbon_density = midmean * (1 + bgbd_factor + 0.11) * 0.47 if not np.isnan(midmean) else np.nan
        se = np.nan if n_mid <= 1 else np.std(mid_vals, ddof=1) / np.sqrt(n_mid)
        
        results.append((land_use_class, carbon_density, n0, se))

    # Save output
    output = pd.DataFrame(
        results,
        columns=["land use class", "carbon density", "n", "se"]
    )
    
    if output_ext == '.csv':
        output.to_csv(output_file, index=False)
    else:  # .parquet
        output.to_parquet(output_file)

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate carbon density using GEDI and JRC data")
    parser.add_argument("--jrc", type=str, required=True, dest="jrc_directory_path", help="Location of JRC tiles.")
    parser.add_argument("--gedi", type=str, required=True, dest="gedi_data_file", help="Location of filtered GEDI data.")
    parser.add_argument("--output", type=str, required=True, dest="output_filename", help="Output file name.")
    args = parser.parse_args()

    generate_carbon_density(args.jrc_directory_path, args.gedi_data_file, args.output_filename)

if __name__ == "__main__":
    main()
