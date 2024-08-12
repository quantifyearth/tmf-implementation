#!/bin/bash

#run with command: scripts/tmfpython.sh -p 1113 -t 2010 ...
#p: project ID 
#t: year of project start (t0)
#e: evaluation year (default: 2022)
#r: whether to run an ex-post evaluation and knit the results in an R notebook (true/false, default: false).
#a: whether to run an ex-ante evaluation and knit the results in an R notebook (true/false, default: false).

#NB running evaluations requires the evaluations code

# Check which branch is currently checked out
current_branch=$(git rev-parse --abbrev-ref HEAD)

set -e

############ DEFAULTS ###############

input_dir="/maps/aew85/projects"
output_dir="/maps/aew85/tmf_pipe_out"
eval_year=2022
ex_post=false
ex_ante=false

#####################################

function display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -p <project>        Project name"
    echo "  -t <t0>             Start year"
    echo "  -e <year>           Evaluation year"
    echo "  -r <ex_post>        Knit ex post evaluation? (true/false)"
    echo "  -a <ex_ante>        Knit ex ante evaluation? (true/false)"
    echo "  -h                  Display this help message"
    echo
    echo "Example:"
    echo "  $0 -p 'gola' -t 2012 -e 2021 -r true -a true"
}

# Parse arguments
while getopts "p:t:e:r:a:h" flag
do
    case "${flag}" in
        p) proj=${OPTARG};;
        t) t0=${OPTARG};;
        e) eval_year=${OPTARG};;
        r) ex_post=${OPTARG};;
        a) ex_ante=${OPTARG};;
        h) display_help; exit 0;;
        *) echo "Invalid option: -${OPTARG}" >&2; display_help; exit 1;;
    esac
done

echo "Project: $proj"
echo "t0: $t0"
echo "Evaluation year: $eval_year"
echo "Ex-post evaluation: $ex_post"
echo "Ex-ante evaluation: $ex_ante"

if [ $# -eq 0 ]; then
    display_help
    exit 1
fi

# Make project output directory
mkdir -p "${output_dir}/${proj}"
echo "--Folder created.--"

#Make buffer
tmfpython3 -m methods.inputs.generate_boundary --project "${input_dir}/${proj}.geojson" --output "${output_dir}/${proj}/buffer.geojson"
echo "--Buffer created.--"

#Make leakage area
tmfpython3 -m methods.inputs.generate_leakage --project "${input_dir}/${proj}.geojson" --output "${output_dir}/${proj}/leakage.geojson"
echo "--Leakage created.--"

# Get GEDI data
tmfpython3 -m methods.inputs.locate_gedi_data "${output_dir}/${proj}/buffer.geojson" /maps/4C/gedi/granule/info/
tmfpython3 -m methods.inputs.download_gedi_data /maps/4C/gedi/granule/info/* /maps/4C/gedi/granule/
tmfpython3 -m methods.inputs.filter_gedi_data --buffer "${output_dir}/${proj}/buffer.geojson" \
                                        --granules /maps/4C/gedi/granule/ \
                                        --output  "${output_dir}/${proj}/gedi.geojson"
tmfpython3 -m methods.inputs.generate_carbon_density --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
                                                --gedi "${output_dir}/${proj}/gedi.geojson" \
                                                --output "${output_dir}/${proj}/carbon-density.csv"

echo "--GEDI data obtained.--"

#Generate list of overlapping countries
tmfpython3 -m methods.inputs.generate_country_list \
--leakage "${output_dir}/${proj}/leakage.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/country-list.json"
echo "--Country list created.--"

#Generate matching area
tmfpython3 -m methods.inputs.generate_matching_area --project "${input_dir}/${proj}.geojson" \
--countrycodes "${output_dir}/${proj}/country-list.json" \
--countries /maps/4C/osm_boundaries.geojson \
--ecoregions /maps/4C/ecoregions/ecoregions.geojson \
--projects /maps/mwd24/tmf-data/projects \
--output "${output_dir}/${proj}/matching-area.geojson"
echo "--Matching area created.--"

#Download SRTM data
tmfpython3 -m methods.inputs.download_srtm_data --project "${input_dir}/${proj}.geojson" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--zips "${output_dir}/srtm/zip" \
--tifs "${output_dir}/srtm/tif"
echo "--SRTM downloaded.--"

#Generate slopes
tmfpython3 -m methods.inputs.generate_slope --input "${output_dir}/srtm/tif" --output "${output_dir}/slopes"
echo "--Slope created.--"

#Rescale to JRC tiles
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/srtm/tif" \
--output "${output_dir}/rescaled-elevation"
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/slopes" \
--output "${output_dir}/rescaled-slopes"
echo "--JRC rescaled.--"

#Create country raster
tmfpython3 -m methods.inputs.generate_country_raster --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/countries.tif"
echo "--Country raster created.--"

#Matching: calculate set K
tmfpython3 -m methods.matching.calculate_k \
--project "${input_dir}/${proj}.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/k.parquet"
echo "--Set K created.--"

#Matching: calculate set M
tmfpython3 -m methods.matching.find_potential_matches \
--k "${output_dir}/${proj}/k.parquet" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/matches"
tmfpython3 -m methods.matching.build_m_raster \
--rasters_directory "${output_dir}/${proj}/matches" \
--output "${output_dir}/${proj}/matches.tif" \
-j 20
tmfpython3 -m methods.matching.build_m_table \
--raster "${output_dir}/${proj}/matches.tif" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/matches.parquet"
echo "--Set M created.--"

#Matching: find pairs
if [ "$current_branch" == "tws_cluster_find_pairs" ] -o [ "$current_branch" == "aew85_cluster_find_pairs" ]; then
tmfpython3 -m methods.matching.find_pairs \
    --k "${output_dir}/${proj}/k.parquet" \
    --m "${output_dir}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --luc_match True \
    --output "${output_dir}/${proj}/pairs" \
    --seed 42 \
    -j 1
    echo "--Pairs matched.--"
    else
    tmfpython3 -m methods.matching.find_pairs \
    --k "${output_dir}/${proj}/k.parquet" \
    --m "${output_dir}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --output "${output_dir}/${proj}/pairs" \
    --seed 42 \
    -j 1
    echo "--Pairs matched.--"
fi

#Calculate additionality
if [ "$current_branch" == "mwd-check-stopping-criteria" ]; then
    tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${input_dir}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${output_dir}/${proj}/carbon-density.csv" \
    --matches "${output_dir}/${proj}/pairs" \
    --output "${output_dir}/${proj}/additionality.csv" \
    --stopping "${output_dir}/${proj}/stopping.csv"
    echo "--Additionality and stopping criteria calculated.--"
    else if [ "$ex_ante" == "true" ]; then
    else
    tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${input_dir}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${output_dir}/${proj}/carbon-density.csv" \
    --matches "${output_dir}/${proj}/pairs" \
    --output "${output_dir}/${proj}/additionality.csv"
    echo "--Additionality calculated.--"
fi

# Run ex post evaluation
if [ "$ex_post" == "true" ]; then
ep_output_file="${evaluations_dir}/${proj}_ex_post_evaluation.html"
Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_post_evaluation_template.Rmd',output_file='${ep_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}',evaluations_dir='${evaluations_dir}'))"
fi

# Run ex-ante evaluation
if [ "$ex_ante" == "true" ]; then
ea_output_file="${evaluations_dir}/${proj}_ex_ante_evaluation.html"
Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_ante_evaluation_template.Rmd',output_file='${ea_output_file}',params=list(proj='${proj}',t0='${t0}',input_dir='${input_dir}',output_dir='${output_dir}'))"
fi