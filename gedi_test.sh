#!/bin/bash

# Define project IDs, end years, and buffer sizes
projects=("1650")
end_years=("2021" "2022" "2023")
buffer_sizes=("70" "80" "90" "100")

# Base directories
project_dir="/maps/jh2589/projects"
gedi_dir="/maps/jh2589/gedi_files/granule"
output_dir="/maps/jh2589/tmf_pipe_out/gedi_results"


# Loop through each combination of project, buffer size, and end year
for project in "${projects[@]}"; do
  for buffer_size in "${buffer_sizes[@]}"; do
    for end_year in "${end_years[@]}"; do
      
      # Directory Make
      mkdir -p "${output_dir}/${project}/${buffer_size}/${end_year}"
      echo "--Folder created.--"
      echo "${output_dir}/${project}/${buffer_size}/${end_year}"

      # Generate boundary
      tmfpython3 -m methods.special.special_generate_boundary \
        --project "$project_dir/$project.geojson" \
        --buffer_size "${buffer_size}" \
        --output "$output_dir/$project/${buffer_size}/${end_year}/boundary.geojson"
        echo "--Boundary created.--"

      # Locate GEDI data
      tmfpython3 -m methods.special.special_locate_gedi_data \
        --gedi-dir "/maps/jh2589/gedi_files/info/" \
        --buffer "$output_dir/$project/${buffer_size}/${end_year}/boundary.geojson" \
        --output-folder "$output_dir/$project/${buffer_size}/${end_year}/" \
        --start-year 2020 \
        --end-year "$end_year"
        echo "--GEDI located.--"

      # Download GEDI
      tmfpython3 -m methods.special.special_download_gedi_data /maps/jh2589/gedi_files/info/* /maps/jh2589/gedi_files/granule
      echo "--DOWNLOADED:)--"

      # Filter GEDI data
      tmfpython3 -m methods.special.special_filter_gedi_data \
        --granules "/maps/jh2589/gedi_files/granule" \
        --buffer "$output_dir/$project/${buffer_size}/${end_year}/boundary.geojson" \
        --granule-list "$output_dir/$project/${buffer_size}/${end_year}/gedi_names.txt" \
        --output "$output_dir/$project/${buffer_size}/${end_year}/gedi.geojson"
        echo "--GEDI filtered.--"

      # Generate carbon density
      tmfpython3 -m methods.special.special_generate_carbon_density \
        --project_name "$project" \
        --buffer_size "${buffer_size}" \
        --end_year "$end_year" \
        --jrc "/maps/forecol/data/JRC/v1_2022/AnnualChange/tifs" \
        --gedi "$output_dir/$project/${buffer_size}/${end_year}/gedi.geojson" \
        --output "$output_dir/csv/${project}_${buffer_size}_${end_year}.csv"
      echo "--Densities located.--"
    done
  done
done