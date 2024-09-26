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
d=`date +%Y_%m_%d`
name_time=out_"$proj"_"$luc_match"_"$d"_time
name_out=out_"$proj"_"$luc_match"_"$d"_out
name_memory=out_"$proj"_"$luc_match"_"$d"_memory

touch "$name_time".txt
touch "$name_out".txt
touch "$name_memory".txt

if [ "$luc_match" == "True" ]; then
output_dir="/maps/epr26/tmf_pipe_out_luc_t"
else
output_dir="/maps/epr26/tmf_pipe_out_luc_f"
fi
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


#open up specific virtual environment to run the sped-up find_pairs.py
. ./venv/bin/activate

#Matching: find pairs
/bin/time -f "\nMemory used by find_pairs: %M KB" python3 -m methods.matching.find_pairs \
    --k "${output_dir}/${proj}/k.parquet" \
    --m "${output_dir}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --luc_match $luc_match \
    --output "${output_dir}/${proj}/pairs" \
    --seed 42 \
    -j 1 \
    2>&1 | tee -a "$name_out".txt >> "$name_memory".txt
echo "--Pairs matched.--" | tee -a "$name_out".txt
tpairs=`date +%s`
echo Pairs: `expr $tpairs - $tm` seconds. | tee -a "$name_time".txt

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
    2>&1 | tee -a "$name_out".txt >> "$name_memory".txt
    echo "--Additionality and stopping criteria calculated.--" | tee -a "$name_out".txt
    else
    /bin/time -f "\nMemory used by calculate_additionality: %M KB" tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${input_dir}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${output_dir}/${proj}/carbon-density.csv" \
    --matches "${output_dir}/${proj}/pairs" \
    --output "${output_dir}/${proj}/additionality.csv" \
    2>&1 | tee -a "$name_out".txt >> "$name_memory".txt
    echo "--Additionality calculated.--" | tee -a "$name_out".txt
fi
tadditionality=`date +%s`
echo Additionality: `expr $tadditionality - $tpairs` seconds. | tee -a "$name_time".txt

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
echo Total execution time: `expr $end - $start` seconds. | tee -a "$name_time".txt
# echo GEDI located: `expr $tgediloc - $tleakage` seconds.
# echo GEDI downloaded: `expr $tgediload - $tgediloc` seconds.
# echo GEDI filtered: `expr $tgedifilt - $tgediload` seconds.
# echo Carbon: `expr $tcarbon - $tgedifilt` seconds.
# echo Country list: `expr $tcountrylist - $tcarbon` seconds.
# echo Matching area: `expr $tmatcharea - $tcountrylist` seconds.
# echo SRTM: `expr $tsrtm - $tmatcharea` seconds.
# echo Slope: `expr $tslope - $tsrtm` seconds.
# echo JRC tiles: `expr $tjrc - $tslope` seconds.
# echo Country raster: `expr $tcountryraster - $tjrc` seconds.
# echo K set: `expr $tk - $tcountryraster` seconds.
# echo M set: `expr $tm - $tk` seconds.
# echo Pairs: `expr $tpairs - $tm` seconds.
# echo Additionality: `expr $tadditionality - $tpairs` seconds.