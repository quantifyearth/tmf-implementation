#!/bin/bash

#run with command: scripts/tmfpython.sh -i 'maps/aew85/projects' -o '/maps/aew85/tmf_pipe_out' -p 1113 -t 2010 ...
#i: input dir - directory containing project shapefiles
#o: output dir - directory containing pipeline outputs
#p: project name/ID - must match name of shapefile
#t: year of project start (t0)
#e: evaluation year (default: 2022)
#v: verbose - whether to run an ex-ante evaluation and knit the results in an R notebook (true/false, default: false).

#NB running evaluations requires the evaluations code

# Check which branch is currently checked out
#current_branch=$(git rev-parse --abbrev-ref HEAD)

set -e

############ DEFAULTS ###############

input_dir="/maps/epr26/tmf-data/projects"
output_dir="/maps/epr26/tmf_pipe_out_offset"
eval_year=2022
verbose=false

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
    echo "  -v <verbose>        Knit ex ante evaluation as .Rmd? (true/false)"
    echo "  -h                  Display this help message"
    echo "Example:"
    echo "  $0 -i '/maps/aew85/projects' -o '/maps/aew85/tmf_pipe_out -p 1201 -t 2012"
}

# Parse arguments
while getopts "i:o:p:t:e:v:h" flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_dir=${OPTARG};;
        p) proj=${OPTARG};;
        t) t0=${OPTARG};;
        e) eval_year=${OPTARG};;
        r) verbose=${OPTARG};;
        h) display_help; exit 0;;
        *) echo "Invalid option: -${OPTARG}" >&2; display_help; exit 1;;
    esac
done

#start clock
start=`date +%s`
d=`date +%Y_%m_%d`

touch out_"$proj"_"$d".txt

echo "Branch: aew85-ex-ante-offset" | tee -a out_"$proj"_"$d".txt
echo "Input directory: $input_dir" | tee -a out_"$proj"_"$d".txt
echo "Output directory: $output_dir" | tee -a out_"$proj"_"$d".txt
echo "Project: $proj" | tee -a out_"$proj"_"$d".txt
echo "t0: $t0" | tee -a out_"$proj"_"$d".txt
echo "Evaluation year: $eval_year" | tee -a out_"$proj"_"$d".txt
echo "Verbose: $verbose" | tee -a out_"$proj"_"$d".txt

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
tbuffer=`date +%s`
echo Buffer: `expr $tbuffer - $start` seconds. | tee -a out_"$proj"_"$d".txt

#Make leakage area
tmfpython3 -m methods.inputs.generate_leakage --project "${input_dir}/${proj}.geojson" --output "${output_dir}/${proj}/leakage.geojson"
echo "--Leakage created.--"
tleakage=`date +%s`
echo Leakage: `expr $tleakage - $tbuffer` seconds. | tee -a out_"$proj"_"$d".txt

# Get GEDI data
tmfpython3 -m methods.inputs.locate_gedi_data "${output_dir}/${proj}/buffer.geojson" /maps/4C/gedi/granule/info/
echo "--GEDI data located.--"
tgediloc=`date +%s`
echo GEDI located: `expr $tgediloc - $tleakage` seconds. | tee -a out_"$proj"_"$d".txt

tmfpython3 -m methods.inputs.download_gedi_data /maps/4C/gedi/granule/info/* /maps/4C/gedi/granule/
echo "--GEDI data downloaded.--"
tgediload=`date +%s`
echo GEDI downloaded: `expr $tgediload - $tgediloc` seconds. | tee -a out_"$proj"_"$d".txt

tmfpython3 -m methods.inputs.filter_gedi_data --buffer "${output_dir}/${proj}/buffer.geojson" \
                                        --granules /maps/4C/gedi/granule/ \
                                        --output  "${output_dir}/${proj}/gedi.geojson"
echo "--GEDI data filtered.--"
tgedifilt=`date +%s`
echo GEDI filtered: `expr $tgedifilt - $tgediload` seconds. | tee -a out_"$proj"_"$d".txt

tmfpython3 -m methods.inputs.generate_carbon_density --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
                                                --gedi "${output_dir}/${proj}/gedi.geojson" \
                                                --output "${output_dir}/${proj}/carbon-density.csv"

echo "--Carbon density calculated.--"
tcarbon=`date +%s`
echo Carbon density: `expr $tcarbon - $tgedifilt` seconds. | tee -a out_"$proj"_"$d".txt

#Generate list of overlapping countries
tmfpython3 -m methods.inputs.generate_country_list \
--leakage "${output_dir}/${proj}/leakage.geojson" \
--countries /maps/4C/osm_boundaries.geojson \
--output "${output_dir}/${proj}/country-list.json"
echo "--Country list created.--"
tcountrylist=`date +%s`
echo Country list: `expr $tcountrylist - $tcarbon` seconds. | tee -a out_"$proj"_"$d".txt

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

#Matching: find pairs
. ./venv/bin/activate
python3 -m methods.matching.find_pairs \
--k "${output_dir}/${proj}/k.parquet" \
--m "${output_dir}/${proj}/matches.parquet" \
--start_year "$t0" \
--eval_year "$eval_year" \
--luc_match True \
--output "${output_dir}/${proj}/pairs" \
--seed 42 \
-j 1
echo "--Pairs matched.--"
tpairs=`date +%s`
echo Pairs: `expr $tpairs - $tm` seconds. | tee -a out_"$proj"_"$d".txt
deactivate

# Run ex-ante evaluation
if [ "$verbose" == "true" ]; then
evaluations_dir="~/evaluations"
ea_output_file="${evaluations_dir}/${proj}_ex_ante_evaluation.html"
Rscript -e "rmarkdown::render(input='~/evaluations/R/ex_ante_evaluation_template.Rmd',output_file='${ea_output_file}',params=list(proj='${proj}',t0='${t0}',eval_year='${eval_year}',input_dir='${input_dir}',output_dir='${output_dir}'))"
fi


#end clock
end=`date +%s`
echo "$proj was done" | tee -a out_"$proj"_"$d".txt
echo Total execution time was `expr $end - $start` seconds.
echo GEDI located: `expr $tgediloc - $tleakage` seconds.
echo GEDI downloaded: `expr $tgediload - $tgediloc` seconds.
echo GEDI filtered: `expr $tgedifilt - $tgediload` seconds.
echo Carbon density: `expr $tcarbon - $tgedifilt` seconds.
echo Country list: `expr $tcountrylist - $tcarbon` seconds.
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds.
echo SRTM: `expr $tsrtm - $tmatcharea` seconds.
echo Slope: `expr $tslope - $tsrtm` seconds.
echo JRC tiles: `expr $tjrc - $tslope` seconds.
echo Country raster: `expr $tcountryraster - $tjrc` seconds.
echo K set: `expr $tk - $tcountryraster` seconds.
echo M set: `expr $tm - $tk` seconds.
echo Pairs: `expr $tpairs - $tm` seconds.