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
lagged=false
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
    echo "  -l <lagged>            Use time-lagged matching (true/false)"
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
        l) lagged=${OPTARG};;
        r) ex_post=${OPTARG};;
        a) ex_ante=${OPTARG};;
        h) display_help; exit 0;;
        *) echo "Invalid option: -${OPTARG}" >&2; display_help; exit 1;;
    esac
done

#start clock
start=`date +%s`

if [ "$lagged" == "True" ] || [ "$lagged" == "true" ]; then
    output_dir="/maps/epr26/tmf_pipe_out_lagged"
else
    output_dir="/maps/epr26/tmf_pipe_out_luc_t"
fi

d=`date +%Y_%m_%d`
name_out="$output_dir"/out_"$proj"_"$d"
touch "$name_out".txt

echo "Output dir: $output_dir" | tee -a "$name_out".txt
echo "Project: $proj" | tee -a "$name_out".txt
echo "t0: $t0" | tee -a "$name_out".txt
echo "Evaluation year: $eval_year" | tee -a "$name_out".txt
echo "Time-lagged matching: $lagged" | tee -a "$name_out".txt
echo "Ex-post evaluation: $ex_post" | tee -a "$name_out".txt
echo "Ex-ante evaluation: $ex_ante" | tee -a "$name_out".txt


#Calculate additionality
/bin/time -f "\nMemory used by calculate_additionality: %M KB" tmfpython3 -m methods.outputs.calculate_additionality \
--project "${input_dir}/${proj}.geojson" \
--lagged "$lagged" \
--project_start "$t0" \
--evaluation_year "$eval_year" \
--density "${output_dir}/${proj}/carbon-density.csv" \
--matches "${output_dir}/${proj}/pairs" \
--output "${output_dir}/${proj}/additionality.csv" \
2>&1 | tee -a "$name_out".txt
echo "--Additionality calculated.--" | tee -a "$name_out".txt
end=`date +%s`
echo Pairs: `expr $end - $start` seconds. | tee -a "$name_out".txt
