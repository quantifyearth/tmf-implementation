
import pandas as pd
import numpy as np
from pyproj import Proj
from numba import jit
import time
import os

grid_resolution_m = 5000

# Read in the data
m_parquet_filename = '/maps/aew85/tmf_pipe_out/1201/matches.parquet'

m_parquet_dirname = os.path.dirname(m_parquet_filename)

k_pixels = pd.read_parquet(k_parquet_filename)
# k_pixels = pd.read_parquet('/maps/tws36/tmf_pipe_out/1201/k_all.parquet')
m_pixels = pd.read_parquet(m_parquet_filename)

def assign_cells(aeqd_x, aeqd_y):
    grid_x = (aeqd_x // grid_resolution_m).astype(int)
    grid_y = (aeqd_y // grid_resolution_m).astype(int)
    return grid_x, grid_y


# Create the directory if it doesn't exist
output_dir = m_parquet_dirname + '/m_tiles'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the AEQD projection
lat_mean = m_pixels['lat'].mean()
lng_mean = m_pixels['lng'].mean()
aeqd_proj = Proj(proj='aeqd', lat_0=lat_mean, lon_0=lng_mean, datum='WGS84')

# Vectorized transformation using pyproj (no need for a loop)
m_pixels['aeqd_x'], m_pixels['aeqd_y'] = aeqd_proj(m_pixels['lng'].values, m_pixels['lat'].values)
m_pixels['grid_x'], m_pixels['grid_y'] = assign_cells(m_pixels['aeqd_x'], m_pixels['aeqd_y'])

print(m_pixels[['grid_x', 'grid_y']])

# Create a unique identifier for each grid cell
m_pixels['grid_id'] = m_pixels['grid_x'].astype(str) + '_' + m_pixels['grid_y'].astype(str)
# Check the number of unique grid cells
print(f"Number of unique grid cells: {m_pixels['grid_id'].nunique()}")

# # Create a color palette with a unique color for each grid_id
# unique_grid_ids = m_pixels['grid_id'].unique()
# palette = sns.color_palette("husl", len(unique_grid_ids))
# color_dict = dict(zip(unique_grid_ids, palette))

# # Map the grid_id to a color
# m_pixels['color'] = m_pixels['grid_id'].map(color_dict)

# # Plot the points
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10, 8))
# plt.scatter(m_pixels['lng'], m_pixels['lat'], c=m_pixels['color'], s=1, alpha=0.6)
# plt.title('Points Colored by Unique Grid Cell')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.savefig('Figures/Gola_m_grid_cells.png')
# plt.close()

# Loop through each unique grid_id
for grid_id in m_pixels['grid_id'].unique():
    # Filter the data for this grid_id
    grid_data = m_pixels[m_pixels['grid_id'] == grid_id]
    
    # Create a unique file name
    file_name = f"m_tile_{grid_id}.parquet"
    file_path = os.path.join(output_dir, file_name)
    
    # Save the DataFrame as a parquet file
    grid_data.to_parquet(file_path, index=False)
    print(f"Saved {file_name} with {len(grid_data)} records.")


#-----------------------------------------
# Code to move to find pairs
# just testing for now



import random

# Directory containing the Parquet files
input_dir = m_parquet_dirname + '/m_tiles'

# List all Parquet files in the directory
all_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

# Determine the number of files to read (10% of total files)
num_files_to_read = max(1, int(0.1 * len(all_files)))  # Ensure at least 1 file is read

# Randomly select 10% of the files
selected_files = random.sample(all_files, num_files_to_read)

# Read the selected files and store them in a list of DataFrames
dfs = []
for file_name in selected_files:
    file_path = os.path.join(input_dir, file_name)
    df = pd.read_parquet(file_path)
    dfs.append(df)

# Concatenate all the DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Print information about the loaded data
print(f"Loaded {len(dfs)} files with a total of {len(combined_df)} records.")