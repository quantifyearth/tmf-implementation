# source myenv/bin/activate

import pandas as pd
import numpy as np
from numba import njit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Proj, transform
import os
import time
import sys
import faiss

# Read K and M
# Repeat 100 times - Ultimately we would like to remove this step
#   PCA across K and M
#   Divide into clusters - Check how much splitting into clusters speeds things up
#   Sample K to find K_sub
#   Sample M to find M_sub
#   Split M into LUC combinations - How big is each set?
#   For each cluster in K_sub
#       Split cluster in K_sub into categorical combinations - How big is each set?
#       For each categorical combination
#           Sample from the equivalent cluster and categorical combindation in M_sub
#           Find pairs for the categorical combinations in K_sub from the categorical combinations in M_sub
#       RowBind categorical combination sets
#   RowBind cluster sets
#   Save Pairs


# NOTES
# 1. We might need to combine some categorical subsets because at least for comparing validity of the matches
#    because if the subsets are too small the standardised mean differences can be very wrong
# 2. One option would be to combine clusters based on the proximity of the cluster centres. For the LUCs, we might 
#    combine groups that are in the same state when the project begins, even if the LUC history is not the same.
# 3. There is a question of how much supposed additionality is created by each categorical subset? If it is 
#    nothing, it might not matter. If it is substantive then it definitely does matter.



# this function uses loops instead of numpy vector operations
@njit
def loop_match(m_pca, k_pca):
    picked = np.ones((m_pca.shape[0],), dtype=np.bool_)
    fast_matches = np.full(k_pca.shape[0], -1, dtype=np.int32)
    for i in range(0, k_pca.shape[0]):
        min_squared_diff_sum = np.inf
        min_squared_diff_j = -1
        for j in range(0, m_pca.shape[0]):
            if picked[j]:
                squared_diff_sum = np.sum((m_pca[j, :] - k_pca[i, :])**2)
                if squared_diff_sum < min_squared_diff_sum:
                    min_squared_diff_sum = squared_diff_sum
                    min_squared_diff_j = j
        fast_matches[i] = min_squared_diff_j
        picked[min_squared_diff_j] = False
    return fast_matches

### Now try with real numbers

def to_int32(x):
    # Normalize the data to the range 0 to 1
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_data = (x - min_val) / (max_val - min_val)
    
    # Scale the normalized data to the range 0 to 255 for unsigned 8-bit integers
    scaled_data = normalized_data * 255
    
    # Convert to 32-bit integers (0 to 255)
    int32_data = scaled_data.astype(np.int32)
    
    return int32_data

def to_pca_int32(x):
    # Perform PCA and convert to dataframe
    pca = PCA(n_components=min(len(x), len(x.columns)), 
            whiten=False)  # Centering and scaling done by default
    pca_result = pca.fit_transform(x)
    pca_df = pd.DataFrame(pca_result)
    # Convert all columns to int8
    pca_32 = pca_df.apply(to_int32)
    return pca_32


def calculate_smd(group1, group2):
    # Means
    mean1, mean2 = np.mean(group1), np.mean(group2)
    # Standard deviations
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    # Sample sizes
    n1, n2 = len(group1), len(group2)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    # Standardized mean difference
    smd = (mean1 - mean2) / pooled_std
    return smd, mean1, mean2, pooled_std


K_SUB_PROPORTION = 0.01
M_SUB_PROPORTION = 0.1
# Number of clusters
N_CLUSTERS = 9
# Number of iterations for K means fitting
N_ITERATIONS = 100
VERBOSE = True

# Define the start year
t0 = 2012 # READ THIS IN

# Read in the data
boundary = gpd.read_file('/maps/aew85/projects/1201.geojson')

k_pixels = pd.read_parquet('/maps/tws36/tmf_pipe_out/1201/k.parquet')
# k_pixels = pd.read_parquet('/maps/tws36/tmf_pipe_out/1201/k_all.parquet')
m_pixels = pd.read_parquet('/maps/aew85/tmf_pipe_out/1201/matches.parquet')


t0 = 2018
boundary = gpd.read_file('/maps/aew85/projects/ona.geojson')
k_pixels = pd.read_parquet('/maps/aew85/tmf_pipe_out/fastfp_test_ona/k.parquet')
m_pixels = pd.read_parquet('/maps/aew85/tmf_pipe_out/fastfp_test_ona/matches.parquet')

if(m_pixels.shape[0] > (k_pixels.shape[0])):
    m_sub_size = int(k_pixels.shape[0]) # First down sample M as it is ~230 million points    
    m_random_indices = np.random.choice(m_pixels.shape[0], size=m_sub_size, replace=False)
    m_pixels = m_pixels.iloc[m_random_indices]

# # Calculate the central coordinates (centroid)
# central_lat = m_pixels['lat'].mean()
# central_lon = m_pixels['lng'].mean()
# aeqd_proj = f"+proj=aeqd +lat_0={central_lat} +lon_0={central_lon} +datum=WGS84"

# # Convert the DataFrame to a GeoDataFrame
# m_gdf = gpd.GeoDataFrame(m_pixels, geometry=gpd.points_from_xy(m_pixels.lng, m_pixels.lat))
# # Set the original CRS to WGS84 (EPSG:4326)
# m_gdf.set_crs(epsg=4326, inplace=True)

# # Transform the GeoDataFrame to the AEQD projection
# m_gdf_aeqd = m_gdf.to_crs(aeqd_proj)

# # Extract the transformed coordinates
# gdf_aeqd['aeqd_x'] = gdf_aeqd.geometry.x
# gdf_aeqd['aeqd_y'] = gdf_aeqd.geometry.y

# # Define the grid resolution in meters
# grid_resolution_m = 5000  # 5 km

# # Calculate grid cell indices
# gdf_aeqd['grid_x'] = (gdf_aeqd['aeqd_x'] // grid_resolution_m).astype(int)
# gdf_aeqd['grid_y'] = (gdf_aeqd['aeqd_y'] // grid_resolution_m).astype(int)

# # Print the first few rows to verify
# print(gdf_aeqd.head())


# concat m and k
km_pixels = pd.concat([k_pixels.assign(trt='trt', ID=range(0, len(k_pixels))),
                       m_pixels.assign(trt='ctrl', ID=range(0, len(m_pixels)))], ignore_index=True)

# Select columns (excluding 'x', 'y', 'lat', 'lng', 'country', 'ecoregion', 'trt', and those starting with 'luc')
exclude_columns = ['ID', 'x', 'y', 'lat', 'lng', 'country', 'ecoregion', 'trt']
exclude_columns += [col for col in km_pixels.columns if col.startswith('luc')]

# Extract only the continuous columns
continuous_columns = km_pixels.columns.difference(exclude_columns)
km_pixels_selected = km_pixels[continuous_columns]

# PCA transform and conversion to 32 bit ints
km_pca = to_pca_int32(km_pixels_selected)

# Looks good
km_pca.head()

#------------------------------------------

# Initialize the KMeans object
kmeans = faiss.Kmeans(d=km_pca.shape[1], k=N_CLUSTERS, niter=N_ITERATIONS, verbose=True)
# Perform clustering
kmeans.train(km_pca)

# Get cluster assignments
km_pixels['cluster'] = kmeans.index.search(km_pca, 1)[1].flatten()

# Frequency distribution in each cluster
cluster_counts = pd.Series(km_pixels['cluster']).value_counts()
if VERBOSE:
    print("Cluster counts:\n", cluster_counts)


# Convert to spatial (simple features)
km_pixels_sf = gpd.GeoDataFrame(
    km_pixels,
    geometry=gpd.points_from_xy(km_pixels['lng'], km_pixels['lat']),
    crs="EPSG:4326"
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot cluster centres

# Get cluster centers
centroids = kmeans.centroids

if VERBOSE:
    # Plot the cluster centers
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x', label='Cluster centers')
    # Add cluster IDs as labels on the plot
    for i, center in enumerate(centroids):
        plt.text(center[0], center[1], str(i), color='red', fontsize=12, weight='bold')
    
    plt.title('K-means Clustering with Faiss')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    plt.savefig('Figures/ona_cluster_centres_faiss_1.png')
    plt.close()  # Close the plot to free up memory

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot clusters as separate facets
if VERBOSE:
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()
    
    clusters = sorted(km_pixels_sf['cluster'].unique())
    
    for i, cluster in enumerate(clusters):
        ax = axes[i]
        cluster_data = km_pixels_sf[km_pixels_sf['cluster'] == cluster]
        cluster_data.plot(ax=ax, color='blue', markersize=0.2)
        boundary.plot(ax=ax, edgecolor='black', facecolor='none')
        ax.set_title(f'Cluster {cluster}')
        ax.set_axis_off()
    
    # Turn off any unused subplots
    for j in range(len(clusters), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('Figures/Ona_cluster_faiss_1_facet.png')
    plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

match_years = [t0-10, t0-5, t0]
match_cats = ["ecoregion", "country", "cluster"] + ["luc_" + str(year) for year in match_years]

# Extract K and M pixels 
k_pixels = km_pixels.loc[km_pixels['trt'] == 'trt']
m_pixels = km_pixels.loc[km_pixels['trt'] == 'ctrl']

# Extract K and M PCA transforms
k_pca = km_pca.loc[km_pixels['trt'] == 'trt'].to_numpy()
m_pca = km_pca.loc[km_pixels['trt'] == 'ctrl'].to_numpy()

k_sub_size = int(k_pixels.shape[0]* K_SUB_PROPORTION)
m_sub_size = int(m_pixels.shape[0] * 1)

# Define indexs for the samples from K and M
k_random_indices = np.random.choice(k_pixels.shape[0], size=k_sub_size, replace=False)
m_random_indices = np.random.choice(m_pixels.shape[0], size=m_sub_size, replace=False)

# Take random samples from K and M pixels
k_sub = k_pixels.iloc[k_random_indices]
m_sub = m_pixels.iloc[m_random_indices]

# Take corresponding random samples from the PCA transformed K and M 
k_sub_pca = k_pca[k_random_indices,:]
m_sub_pca = m_pca[m_random_indices,:]

if VERBOSE:
    # Handy code for displaying the number of counts in each unique category combination
    # In K
    k_combination_counts = k_sub.groupby(match_cats).size().reset_index(name='counts')
    print("k_combination_counts")
    print(k_combination_counts)
    # In M
    m_combination_counts = m_sub.groupby(match_cats).size().reset_index(name='counts')
    print("m_combination_counts")
    print(m_combination_counts)


# Identify the unique combinations of luc columns
k_cat_combinations = k_sub[match_cats].drop_duplicates().sort_values(by=match_cats, ascending=[True] * len(match_cats))

pairs_list = []

start_time = time.time()
for i in range(0, k_cat_combinations.shape[0]):
    # i = 6 # ith element of the unique combinations of the luc time series in k
    # for in range()
    k_cat_comb = k_cat_combinations.iloc[i]
    k_cat = k_sub[(k_sub[match_cats] == k_cat_comb).all(axis=1)]
    k_cat_pca = k_sub_pca[(k_sub[match_cats] == k_cat_comb).all(axis=1)]
    
    # Find the subset in km_pixels that matches this combination
    m_cat = m_sub[(m_sub[match_cats] == k_cat_comb).all(axis=1)]
    m_cat_pca = m_sub_pca[(m_sub[match_cats] == k_cat_comb).all(axis=1)]
    
    if VERBOSE:
        print('ksub_cat:' + str(k_cat.shape[0]))
        print('msub_cat:' + str(m_cat.shape[0]))
    
    # If there is no suitable match for the pre-project luc time series
    # Then it may be preferable to just take the luc state at t0
    # m_luc_comb = m_pixels[(m_pixels[match_luc_years[1:3]] == K_luc_comb[1:3]).all(axis=1)]
    # m_luc_comb = m_pixels[(m_pixels[match_luc_years[2:3]] == K_luc_comb[2:3]).all(axis=1)]
    # For if there are no matches return nothing
    
    if(m_cat.shape[0] < k_cat.shape[0] * 5):
        print("M insufficient for matching. Set VERBOSE to True for more details.")
        continue
    
    matches_index = loop_match(m_cat_pca, k_cat_pca)
    m_cat_matches = m_cat.iloc[matches_index]
    
    # i = 0
    # matched = pd.concat([k_cat.iloc[i], m_cat.iloc[matches[i]]], axis=1, ignore_index=True)
    # matched.columns = ['trt', 'ctrl']
    # matched
    #Looks great!
    columns_to_compare = ['access', 'cpc0_d', 'cpc0_u', 'cpc10_d', 'cpc10_u', 'cpc5_d', 'cpc5_u', 'elevation', 'slope']
    # Calculate SMDs for the specified columns
    smd_results = []
    for column in columns_to_compare:
        smd, mean1, mean2, pooled_std = calculate_smd(k_cat[column], m_cat_matches[column])
        smd_results.append((column, smd, mean1, mean2, pooled_std))
    
    # Convert the results to a DataFrame for better readability
    smd_df = pd.DataFrame(smd_results, columns=['Variable', 'SMD', 'Mean_k_cat', 'Mean_m_cat', 'Pooled_std'])
    
    if VERBOSE:
        # Print the results
        print("categorical combination:")
        print(k_cat_comb)
        # Count how many items in 'column1' are not equal to the specified integer value
        print("LUC flips in K:")
        (k_cat['luc_2022'] != k_cat_comb['luc_' + str(t0)]).sum()
        print("LUC flips in matches:")
        (m_cat_matches['luc_2022'] != k_cat_comb['luc_' + str(t0)]).sum()
        print("Standardized Mean Differences:")
        print(smd_df)
    
    # Join the pairs into one dataframe:
    k_cat = k_cat.reset_index(drop = True)
    m_cat_matches = m_cat_matches.reset_index(drop = True)  
    pairs_df = pd.concat([k_cat.add_prefix('k_'), m_cat_matches.add_prefix('s_')], axis=1)
    
    # Append the resulting DataFrame to the list
    pairs_list.append(pairs_df)

# Combine all the DataFrames in the list into a single DataFrame
combined_pairs = pd.concat(pairs_list, ignore_index=True)

end_time = time.time()
elapsed_time = end_time - start_time
if VERBOSE:
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

columns_to_compare = ['access', 'cpc0_d', 'cpc0_u', 'cpc10_d', 'cpc10_u', 'cpc5_d', 'cpc5_u', 'elevation', 'slope']
# Calculate SMDs for the specified columns
smd_results = []
for column in columns_to_compare:
    smd, mean1, mean2, pooled_std = calculate_smd(combined_pairs['k_' + column], combined_pairs['s_' + column])
    smd_results.append((column, smd, mean1, mean2, pooled_std))

# Convert the results to a DataFrame for better readability
smd_df = pd.DataFrame(smd_results, columns=['Variable', 'SMD', 'Mean_k_cat', 'Mean_m_cat', 'Pooled_std'])
print(smd_df)


