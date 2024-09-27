#!/bin/bash

#run with command: scripts/tmfpython.sh -p 1113 -t 2010 ...
#run ./tmfpython.sh -p '1532' -t 2012  2>&1 | tee out_1532_2024_08_06_all.txt to save all output
#p: project ID 
#t: year of project start (t0)
#e: evaluation year (default: 2022)
#r: whether to run an ex-post evaluation and knit the results in an R notebook (true/false, default: false).
#a: whether to run an ex-ante evaluation and knit the results in an R notebook (true/false, default: false).

#NB running evaluations requires the evaluations code
proj="as12"
luc_match="True"
d=2022

if [ "$luc_match" == "True" ]; then
output_dir="/maps/epr26/tmf_pipe_out_luc_t"
else
output_dir="/maps/epr26/tmf_pipe_out_luc_f"
fi

name_out="$output_dir"/out_"$proj"_"$luc_match"_"$d"_out

echo "Debug: proj=$proj, luc_match=$luc_match, d=$d"
echo "Output directory: $output_dir"
echo "Log file name: $name_out.txt"