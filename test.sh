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

if [ "$luc_match" == "True" ]; then
    if [ "$current_branch" == "epr26-forecast-time-offset" ]; then
        output_dir="/maps/epr26/tmf_pipe_out_offset"
    else
        output_dir="/maps/epr26/tmf_pipe_out"
    fi
else
    output_dir="/maps/epr26/tmf_pipe_out_luc_f"
fi

d=`date +%Y_%m_%d`
name_out="$output_dir"/out_"$proj"_"$luc_match"_"$d"_block
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

/bin/time -f "\nMemory used by build_m_table: %M KB" tmfpython3 -m methods.matching.build_m_table \
--raster "${output_dir}/${proj}/block_baseline.tif" \
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
--output "${output_dir}/${proj}/block_baseline.parquet" \
2>&1 | tee -a "$name_out".txt
echo "--Set M created.--" | tee -a "$name_out".txt
tmtable=`date +%s`

#end clock
end=`date +%s`
echo "$proj was done" | tee -a "$name_out".txt
echo Block baseline table: `expr $tmtable - $start` seconds. | tee -a "$name_out".txt
