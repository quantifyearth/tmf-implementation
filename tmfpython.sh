#!/bin/bash

#run with command: scripts/tmfpython.sh -p 1113 -t 2010 ...
#run ./tmfpython.sh -p '1532' -t 2012  2>&1 | tee out_1532_2024_08_06_all.txt to save all output
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
eval_year=2022
luc_match=true
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
    echo "  -l <luc_match>      Use exact LUC match in find_pairs (true/false)"
    echo "  -r <ex_post>        Knit ex post evaluation? (true/false)"
    echo "  -a <ex_ante>        Knit ex ante evaluation? (true/false)"
    echo "  -h                  Display this help message"
    echo
    echo "Example:"
    echo "  $0 -p 'gola' -t 2012 -e 2021 -r true -a true"
}

# Parse arguments
while getopts "p:t:e:l:r:a:h" flag
do
    case "${flag}" in
        p) proj=${OPTARG};;
        t) t0=${OPTARG};;
        e) eval_year=${OPTARG};;
        l) luc_match=${OPTARG};;
        r) ex_post=${OPTARG};;
        a) ex_ante=${OPTARG};;
        h) display_help; exit 0;;
        *) echo "Invalid option: -${OPTARG}" >&2; display_help; exit 1;;
    esac
done

#start clock
start=`date +%s`

if [ "$current_branch" == "epr26-forecast-time-offset" ]; then
        output_dir="/maps/epr26/tmf_pipe_out_offset_new"
else
    if [ "$luc_match" == "True" ]; then
        output_dir="/maps/epr26/tmf_pipe_out_luc_t"
    else
        output_dir="/maps/epr26/tmf_pipe_out_luc_f"
    fi
fi

d=`date +%Y_%m_%d`
name_out="$output_dir"/out_"$proj"_"$luc_match"_"$d"
touch "$name_out".txt

echo "Output dir: $output_dir" | tee -a "$name_out".txt
echo "Project: $proj" | tee -a "$name_out".txt
echo "t0: $t0" | tee -a "$name_out".txt
echo "Evaluation year: $eval_year" | tee -a "$name_out".txt
echo "LUC match: $luc_match" | tee -a "$name_out".txt
echo "Ex-post evaluation: $ex_post" | tee -a "$name_out".txt
echo "Ex-ante evaluation: $ex_ante" | tee -a "$name_out".txt

if [ "$ex_post" == "true" ]; then
    evaluations_dir="~/evaluations"
    ep_output_file="${evaluations_dir}/${proj}_ex_post_evaluation.html"
    Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_post_evaluation_template.Rmd',output_file='${ep_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}',evaluations_dir='${evaluations_dir}'))"
fi


# Make project output directory
mkdir -p "${output_dir}/${proj}"
echo "--Folder created.--" | tee -a "$name_out".txt

#Make buffer
/bin/time -f "\nMemory used by generate_boundary: %M KB" tmfpython3 -m methods.inputs.generate_boundary \
--project "${input_dir}/${proj}.geojson" \
--output "${output_dir}/${proj}/buffer.geojson" \
2>&1 | tee -a "$name_out".txt
echo "--Buffer created.--" | tee -a "$name_out".txt
tbuffer=`date +%s`
echo Buffer: `expr $tbuffer - $start` seconds. | tee -a "$name_out".txt

#Make leakage area
/bin/time -f "\nMemory used by generate_leakage: %M KB" tmfpython3 -m methods.inputs.generate_leakage \
--project "${input_dir}/${proj}.geojson" \
--output "${output_dir}/${proj}/leakage.geojson" \
2>&1 | tee -a "$name_out".txt
echo "--Leakage created.--" | tee -a "$name_out".txt
tleakage=`date +%s`
echo Leakage: `expr $tleakage - $tbuffer` seconds. | tee -a "$name_out".txt

# Get GEDI data
/bin/time -f "\nMemory used by locate_gedi_data: %M KB" tmfpython3 -m methods.inputs.locate_gedi_data \
--granules /maps/4C/gedi/granule/info/ \
--buffer "${output_dir}/${proj}/buffer.geojson" \
--output "${output_dir}/${proj}/gedi_names.csv" \
2>&1 | tee -a "$name_out".txt
echo "--GEDI data located.--" | tee -a "$name_out".txt
tgediloc=`date +%s`
echo GEDI located: `expr $tgediloc - $tleakage` seconds. | tee -a "$name_out".txt

/bin/time -f "\nMemory used by download_gedi_data: %M KB" tmfpython3 -m methods.inputs.download_gedi_data \
/maps/4C/gedi/granule/info/* /maps/4C/gedi/granule/ \
2>&1 | tee -a "$name_out".txt
echo "--GEDI data downloaded.--" | tee -a "$name_out".txt
tgediload=`date +%s`
echo GEDI downloaded: `expr $tgediload - $tgediloc` seconds. | tee -a "$name_out".txt

/bin/time -f "\nMemory used by filter_gedi_data: %M KB" tmfpython3 -m methods.inputs.filter_gedi_data \
--granules /maps/4C/gedi/granule/ \
--buffer "${output_dir}/${proj}/buffer.geojson" \
--csv "${output_dir}/${proj}/gedi_names.csv" \
--output  "${output_dir}/${proj}/gedi.geojson" \
2>&1 | tee -a "$name_out".txt
echo "--GEDI data filtered.--" | tee -a "$name_out".txt
tgedifilt=`date +%s`
echo GEDI filtered: `expr $tgedifilt - $tgediload` seconds. | tee -a "$name_out".txt


/bin/time -f "\nMemory used by generate_carbon_density: %M KB" tmfpython3 -m methods.inputs.generate_carbon_density \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--gedi "${output_dir}/${proj}/gedi.geojson" \
--output "${output_dir}/${proj}/carbon-density.csv" \
2>&1 | tee -a "$name_out".txt
echo "--Carbon density calculated.--" | tee -a "$name_out".txt
tcarbon=`date +%s`
echo Carbon density: `expr $tcarbon - $tgedifilt` seconds. | tee -a "$name_out".txt

#Generate list of overlapping countries
/bin/time -f "\nMemory used by generate_country_list: %M KB" tmfpython3 -m methods.inputs.generate_country_list \
--leakage "${output_dir}/${proj}/leakage.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/country-list.json" \
2>&1 | tee -a "$name_out".txt
echo "--Country list created.--" | tee -a "$name_out".txt
tcountrylist=`date +%s`
echo Country list: `expr $tcountrylist - $tcarbon` seconds. | tee -a "$name_out".txt

#Generate matching area
/bin/time -f "\nMemory used by generate_matching_area: %M KB" tmfpython3 -m methods.inputs.generate_matching_area \
--project "${input_dir}/${proj}.geojson" \
--countrycodes "${output_dir}/${proj}/country-list.json" \
--countries /maps/4C/osm_boundaries.geojson \
--ecoregions /maps/4C/ecoregions/ecoregions.geojson \
--projects /maps/mwd24/tmf-data/projects \
--output "${output_dir}/${proj}/matching-area.geojson" \
2>&1 | tee -a "$name_out".txt
echo "--Matching area created.--" | tee -a "$name_out".txt
tmatcharea=`date +%s`
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds. | tee -a "$name_out".txt

#Download SRTM data
/bin/time -f "\nMemory used by download_srtm_data: %M KB" tmfpython3 -m methods.inputs.download_srtm_data \
--project "${input_dir}/${proj}.geojson" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--zips "${output_dir}/srtm/zip" \
--tifs "${output_dir}/srtm/tif" \
2>&1 | tee -a "$name_out".txt
echo "--SRTM downloaded.--" | tee -a "$name_out".txt
tsrtm=`date +%s`
echo SRTM: `expr $tsrtm - $tmatcharea` seconds. | tee -a "$name_out".txt

#Generate slopes
/bin/time -f "\nMemory used by generate_slope: %M KB" tmfpython3 -m methods.inputs.generate_slope \
--input "${output_dir}/srtm/tif" \
--output "${output_dir}/slopes" \
2>&1 | tee -a "$name_out".txt
echo "--Slope created.--" | tee -a "$name_out".txt
tslope=`date +%s`
echo Slope: `expr $tslope - $tsrtm` seconds. | tee -a "$name_out".txt

#Rescale to JRC tiles
/bin/time -f "\nMemory used by rescale_tiles_to_jrc: %M KB" tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/srtm/tif" \
--output "${output_dir}/rescaled-elevation" \
2>&1 | tee -a "$name_out".txt
/bin/time -f "\nMemory used by rescale_tiles_to_jrc: %M KB" tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles "${output_dir}/slopes" \
--output "${output_dir}/rescaled-slopes" \
2>&1 | tee -a "$name_out".txt
echo "--JRC rescaled.--" | tee -a "$name_out".txt
tjrc=`date +%s`
echo JRC tiles: `expr $tjrc - $tslope` seconds. | tee -a "$name_out".txt

#Create country raster
/bin/time -f "\nMemory used by generate_country_raster: %M KB" tmfpython3 -m methods.inputs.generate_country_raster \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/countries.tif" \
2>&1 | tee -a "$name_out".txt
echo "--Country raster created.--" | tee -a "$name_out".txt
tcountryraster=`date +%s`
echo Country raster: `expr $tcountryraster - $tjrc` seconds. | tee -a "$name_out".txt

#Matching: calculate set K
/bin/time -f "\nMemory used by calculate_k: %M KB" tmfpython3 -m methods.matching.calculate_k \
--project "${input_dir}/${proj}.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access_walking/ \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/k.parquet" \
2>&1 | tee -a "$name_out".txt
echo "--Set K created.--" | tee -a "$name_out".txt
tkset=`date +%s`
echo K set: `expr $tkset - $tcountryraster` seconds. | tee -a "$name_out".txt

#Matching: calculate set M
/bin/time -f "\nMemory used by find_potential_matches: %M KB" tmfpython3 -m methods.matching.find_potential_matches \
--k "${output_dir}/${proj}/k.parquet" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access_walking/ \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/matches" \
2>&1 | tee -a "$name_out".txt
/bin/time -f "\nMemory used by build_m_raster: %M KB" tmfpython3 -m methods.matching.build_m_raster \
--rasters_directory "${output_dir}/${proj}/matches" \
--output "${output_dir}/${proj}/matches.tif" \
-j 1 \
2>&1 | tee -a "$name_out".txt
/bin/time -f "\nMemory used by build_m_table: %M KB" tmfpython3 -m methods.matching.build_m_table \
--raster "${output_dir}/${proj}/matches.tif" \
--matching "${output_dir}/${proj}/matching-area.geojson" \
--start_year "$t0" \
--evaluation_year "$eval_year" \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation "${output_dir}/rescaled-elevation" \
--slope "${output_dir}/rescaled-slopes" \
--access /maps/4C/access_walking/ \
--countries-raster "${output_dir}/${proj}/countries.tif" \
--output "${output_dir}/${proj}/matches.parquet" \
2>&1 | tee -a "$name_out".txt
echo "--Set M created.--" | tee -a "$name_out".txt
tmset=`date +%s`
echo M set: `expr $tmset - $tkset` seconds. | tee -a "$name_out".txt

#open up specific virtual environment to run the sped-up find_pairs.py
. ./venv/bin/activate

#Matching: find pairs
/bin/time -f "\nMemory used by find_pairs: %M KB" python3 -m methods.matching.find_pairs \
    --k "${output_dir}/${proj}/k.parquet" \
    --m "${output_dir}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --eval_year "$eval_year" \
    --luc_match $luc_match \
    --output "${output_dir}/${proj}/pairs" \
    --seed 42 \
    -j 1 \
    2>&1 | tee -a "$name_out".txt
echo "--Pairs matched.--" | tee -a "$name_out".txt
tpairs=`date +%s`
echo Pairs: `expr $tpairs - $tmset` seconds. | tee -a "$name_out".txt

#switch back
deactivate

#Calculate additionality
if [ "$current_branch" == "mwd-check-stopping-criteria" ]; then
    /bin/time -f "\nMemory used by calculate_additionality: %M KB" tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${input_dir}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${output_dir}/${proj}/carbon-density.csv" \
    --matches "${output_dir}/${proj}/pairs" \
    --output "${output_dir}/${proj}/additionality.csv" \
    --stopping "${output_dir}/${proj}/stopping.csv" \
    2>&1 | tee -a "$name_out".txt
    echo "--Additionality and stopping criteria calculated.--" | tee -a "$name_out".txt
    else
    /bin/time -f "\nMemory used by calculate_additionality: %M KB" tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${input_dir}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${output_dir}/${proj}/carbon-density.csv" \
    --matches "${output_dir}/${proj}/pairs" \
    --output "${output_dir}/${proj}/additionality.csv" \
    2>&1 | tee -a "$name_out".txt
    echo "--Additionality calculated.--" | tee -a "$name_out".txt
fi
tadditionality=`date +%s`
echo Additionality: `expr $tadditionality - $tpairs` seconds. | tee -a "$name_out".txt

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
echo "$proj was done" | tee -a "$name_out".txt
echo Total execution time: `expr $end - $start` seconds. | tee -a "$name_out".txt
echo GEDI located: `expr $tgediloc - $tleakage` seconds. | tee -a "$name_out".txt
echo GEDI downloaded: `expr $tgediload - $tgediloc` seconds. | tee -a "$name_out".txt
echo GEDI filtered: `expr $tgedifilt - $tgediload` seconds. | tee -a "$name_out".txt
echo Carbon: `expr $tcarbon - $tgedifilt` seconds. | tee -a "$name_out".txt
echo Country list: `expr $tcountrylist - $tcarbon` seconds. | tee -a "$name_out".txt
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds. | tee -a "$name_out".txt
echo SRTM: `expr $tsrtm - $tmatcharea` seconds. | tee -a "$name_out".txt
echo Slope: `expr $tslope - $tsrtm` seconds. | tee -a "$name_out".txt
echo JRC tiles: `expr $tjrc - $tslope` seconds. | tee -a "$name_out".txt
echo Country raster: `expr $tcountryraster - $tjrc` seconds. | tee -a "$name_out".txt
echo K set: `expr $tkset - $tcountryraster` seconds. | tee -a "$name_out".txt
echo M set: `expr $tmset - $tkset` seconds. | tee -a "$name_out".txt
echo Pairs: `expr $tpairs - $tmset` seconds. | tee -a "$name_out".txt
echo Additionality: `expr $tadditionality - $tpairs` seconds. | tee -a "$name_out".txt