
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

touch out_"$proj"_"$d".txt

# Make project output directory
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