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
        output_dir="/maps/epr26/tmf_pipe_out_offset"
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
echo Pairs: `expr $tpairs - $start` seconds. | tee -a "$name_out".txt

#switch back
deactivate


#end clock
end=`date +%s`
echo "$proj was done" | tee -a "$name_out".txt
echo Total execution time: `expr $tpairs - $start` seconds. | tee -a "$name_out".txt