#!/bin/bash

#run with command: scripts/tmfpython.sh -i '/maps/aew85/projects' -o '/maps/aew85/tmf_pipe_out' -p 1113 -t 2010 ...
#i: input dir - directory containing project shapefiles
#o: output dir - directory containing pipeline outputs
#p: project name/ID - must match name of shapefile
#t: year of project start (t0)
#e: evaluation year (default: 2022)
#r: report - whether to run an ex-post evaluation and knit the results in an R notebook (true/false, default: false).

#NB running evaluations requires the evaluations code

# Check which branch is currently checked out
branch=$(git rev-parse --abbrev-ref HEAD)

set -e

############ DEFAULTS ###############

input_dir=""
output_dir=""
eval_year=2022
report=false

#####################################

function display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -i <input_dir>      Input directory"
    echo "  -o <output_dir>     Output directory"
    echo "  -p <proj>           Project name"
    echo "  -t <t0>             Start year"
    echo "  -e <year>           Evaluation year"
    echo "  -r <report>         Knit ex post evaluation as .Rmd? (true/false)"
    echo "  -h                  Display this help message"
    echo "Example:"
    echo "  $0 -i '/maps/aew85/projects' -o '/maps/aew85/tmf_pipe_out -p 1201 -t 2012"
}

# Parse arguments
while getopts "i:o:p:t:e:r:h" flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_dir=${OPTARG};;
        p) proj=${OPTARG};;
        t) t0=${OPTARG};;
        e) eval_year=${OPTARG};;
        r) report=${OPTARG};;
        a) ex_ante=${OPTARG};;
        h) display_help; exit 0;;
        *) echo "Invalid option: -${OPTARG}" >&2; display_help; exit 1;;
    esac
done

echo "Input directory: $input_dir"
echo "Output directory: $output_dir"
echo "Project: $proj"
echo "t0: $t0"
echo "Evaluation year: $eval_year"
echo "Create report: $report"

if [ $# -eq 0 ]; then
    display_help
    exit 1
fi

# Make project output folder
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
tmfpython3 -m methods.matching.find_pairs \
--k "${output_dir}/${proj}/k.parquet" \
--m "${output_dir}/${proj}/matches.parquet" \
--start_year "$t0" \
--output "${output_dir}/${proj}/pairs" \
--seed 42 \
-j 1 
echo "--Pairs matched.--"

#Calculate additionality
tmfpython3 -m methods.outputs.calculate_additionality \
--project "${input_dir}/${proj}.geojson" \
--project_start "$t0" \
--evaluation_year "$eval_year" \
--density "${output_dir}/${proj}/carbon-density.csv" \
--matches "${output_dir}/${proj}/pairs" \
--output "${output_dir}/${proj}/additionality.csv"
echo "--Additionality calculated.--"

# Knit report file
if [ "$report" == "true" ]; then
    report_output_file="${output_dir}/${proj}_report.html"
    Rscript -e "rmarkdown::render(input='./scripts/pipeline_results.Rmd',output_file='${report_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}',branch='${branch}))"
fi
