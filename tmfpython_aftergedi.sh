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

input_dir="/maps/epr26/tmf-data/projects"
output_dir="/maps/epr26/tmf_pipe_out_tws"
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

#rm /maps/epr26/tmf_pipe_out/srtm/tif/srtm_61_10.tif

#start clock
start=`date +%s`
d=`date +%Y_%m_%d`

#Generate list of overlapping countries
tmfpython3 -m methods.inputs.generate_country_list \
--leakage "${output_dir}/${proj}/leakage.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/country-list.json"
echo "--Country list created.--"
tcountrylist=`date +%s`
echo Country list: `expr $tcountrylist - $start` seconds. | tee -a out_"$proj"_"$d".txt

#Generate matching area
tmfpython3 -m methods.inputs.generate_matching_area --project "${input_dir}/${proj}.geojson" \
--countrycodes "${output_dir}/${proj}/country-list.json" \
--countries /maps/4C/osm_boundaries.geojson \
--ecoregions /maps/4C/ecoregions/ecoregions.geojson \
--projects /maps/mwd24/tmf-data/projects \
--output "${output_dir}/${proj}/matching-area.geojson"
echo "--Matching area created.--"
tmatcharea=`date +%s`
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds. | tee -a out_"$proj"_"$d".txt

#Download SRTM data
tmfpython3 -m methods.inputs.download_srtm_data --project "${input_dir}/${proj}.geojson" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--zips "${output_dir}/srtm/zip" \
--tifs "${output_dir}/srtm/tif"
echo "--SRTM downloaded.--"
tsrtm=`date +%s`
echo SRTM: `expr $tsrtm - $tmatcharea` seconds. | tee -a out_"$proj"_"$d".txt

#Generate slopes
tmfpython3 -m methods.inputs.generate_slope --input "${output_dir}/srtm/tif" --output "${output_dir}/slopes"
echo "--Slope created.--"
tslope=`date +%s`
echo Slope: `expr $tslope - $tsrtm` seconds. | tee -a out_"$proj"_"$d".txt

#Rescale to JRC tiles
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/srtm/tif" \
--output "${output_dir}/rescaled-elevation"
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/slopes" \
--output "${output_dir}/rescaled-slopes"
echo "--JRC rescaled.--"
tjrc=`date +%s`
echo JRC tiles: `expr $tjrc - $tslope` seconds. | tee -a out_"$proj"_"$d".txt

#Create country raster
tmfpython3 -m methods.inputs.generate_country_raster --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/countries.tif"
echo "--Country raster created.--"
tcountryraster=`date +%s`
echo Country raster: `expr $tcountryraster - $tjrc` seconds. | tee -a out_"$proj"_"$d".txt

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
tk=`date +%s`
echo K set: `expr $tk - $tcountryraster` seconds. | tee -a out_"$proj"_"$d".txt

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
tm=`date +%s`
echo M set: `expr $tm - $tk` seconds. | tee -a out_"$proj"_"$d".txt

. ./venv/bin/activate

#Matching: find pairs
python3 -m methods.matching.find_pairs \
    --k "${output_dir}/${proj}/k.parquet" \
    --m "${output_dir}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --luc_match False \
    --output "${output_dir}/${proj}/pairs" \
    --seed 42 \
    -j 1
    echo "--Pairs matched.--"
tpairs=`date +%s`
echo Pairs: `expr $tpairs - $tm` seconds. | tee -a out_"$proj"_"$d".txt

deactivate

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
tadditionality=`date +%s`
echo Additionality: `expr $tadditionality - $tpairs` seconds. | tee -a out_"$proj"_"$d".txt

# Run ex post evaluation
if [ "$ex_post" == "true" ]; then
evaluations_dir="~/evaluations"
ep_output_file="${evaluations_dir}/${proj}_ex_post_evaluation.html"
Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_post_evaluation_template.Rmd',output_file='${ep_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}',evaluations_dir='${evaluations_dir}'))"
fi

# Run ex-ante evaluation
if [ "$ex_ante" == "true" ]; then
evaluations_dir="~/evaluations"
ea_output_file="${evaluations_dir}/${proj}_ex_ante_evaluation.html"
Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_ante_evaluation_template.Rmd',output_file='${ea_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}',evaluations_dir='${evaluations_dir}'))"
fi


#end clock
end=`date +%s`
echo "$proj was done" | tee -a out_"$proj"_"$d".txt
echo "t0 was $t0" | tee -a out_"$proj"_"$d".txt
echo Total execution time was `expr $end - $start` seconds.
echo Buffer: `expr $tbuffer - $start` seconds.
echo Leakage: `expr $tleakage - $tbuffer` seconds.
echo GEDI: `expr $tgedi - $tleakage` seconds.
echo Country list: `expr $tcountrylist - $tgedi` seconds.
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds.
echo SRTM: `expr $tsrtm - $tmatcharea` seconds.
echo Slope: `expr $tslope - $tsrtm` seconds.
echo JRC tiles: `expr $tjrc - $tslope` seconds.
echo Country raster: `expr $tcountryraster - $tjrc` seconds.
echo K set: `expr $tk - $tcountryraster` seconds.
echo M set: `expr $tm - $tk` seconds.
echo Pairs: `expr $tpairs - $tm` seconds.
echo Additionality: `expr $tadditionality - $tpairs` seconds.