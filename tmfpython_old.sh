#!/bin/bash

#run with command: ./tmfpython.sh -p 1113 -t 2010
#p: project ID
#t: year of project start (t0)

while getopts p:t: flag
do
    case "${flag}" in
        p) proj=${OPTARG};;
        t) t0=${OPTARG};;
    esac
done
echo "Project: $proj"
echo "t0: $t0"

#start clock
start=`date +%s`
d=`date +%Y_%m_%d`

touch out_"$proj"_"$d".txt

#The shapefiles are stored in https://github.com/carboncredits/tmf-data/: I have cloned the repo to /maps/epr26/tmf-data/projects
mkdir /maps/epr26/tmf_pipe_out/"$proj"
echo "--Folder created.--"

#Make buffer
tmfpython3 -m methods.inputs.generate_boundary --project /maps/epr26/tmf-data/projects/"$proj".geojson --output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"buffer.geojson
echo "--Buffer created.--"
tbuffer=`date +%s`

#Make leakage area
tmfpython3 -m methods.inputs.generate_leakage --project /maps/epr26/tmf-data/projects/"$proj".geojson --output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"leakage.geojson
echo "--Leakage created.--"
tleakage=`date +%s`

#Make LUC tiles
tmfpython3 -m methods.inputs.generate_luc_layer --buffer /maps/epr26/tmf_pipe_out/"$proj"/"$proj"buffer.geojson --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs --output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"luc.tif
echo "--LUC created.--"
tluc=`date +%s`

#For the GEDI data section, if it's a project that Tom is already running (i.e., in the carboncredits/tmf-data repo), the first two lines don't need to be run
#The data already has been downloaded and ingested into POSTGIS, you just need to generate AGB data layers

#Generate AGB layers
tmfpython3 -m methods.inputs.generate_carbon_density --buffer /maps/epr26/tmf_pipe_out/"$proj"/"$proj"buffer.geojson --luc /maps/epr26/tmf_pipe_out/"$proj"/"$proj"luc.tif --output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"carbon-density.csv
echo "--ACD created.--"
tagb=`date +%s`

#Generate list of overlapping countries
tmfpython3 -m methods.inputs.generate_country_list \
--leakage /maps/epr26/tmf_pipe_out/"$proj"/"$proj"leakage.geojson \
--countries /maps/4C/osm_boundaries.geojson \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"country-list.json
echo "--Country list created.--"
tcountrylist=`date +%s`

#Generate matching area
tmfpython3 -m methods.inputs.generate_matching_area --project /maps/epr26/tmf-data/projects/"$proj".geojson \
--countrycodes /maps/epr26/tmf_pipe_out/"$proj"/"$proj"country-list.json \
--countries /maps/4C/osm_boundaries.geojson \
--ecoregions /maps/4C/ecoregions/ecoregions.geojson \
--projects /maps/epr26/tmf-data/projects \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matching-area.geojson
echo "--Matching area created.--"
tmatcharea=`date +%s`

#Download SRTM data
tmfpython3 -m methods.inputs.download_srtm_data --project /maps/epr26/tmf-data/projects/"$proj".geojson \
--matching /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matching-area.geojson \
--zips /maps/epr26/tmf_pipe_out/"$proj"/srtm/zip \
--tifs /maps/epr26/tmf_pipe_out/"$proj"/srtm/tif
echo "--SRTM downloaded.--"
tsrtm=`date +%s`

#Generate slopes
tmfpython3 -m methods.inputs.generate_slope --input /maps/epr26/tmf_pipe_out/"$proj"/srtm/tif --output /maps/epr26/tmf_pipe_out/"$proj"/slopes
echo "--Slope created.--"
tslope=`date +%s`

#Rescale to JRC tiles
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles /maps/epr26/tmf_pipe_out/"$proj"/srtm/tif \
--output /maps/epr26/tmf_pipe_out/"$proj"/rescaled-elevation
tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--tiles /maps/epr26/tmf_pipe_out/"$proj"/slopes \
--output /maps/epr26/tmf_pipe_out/"$proj"/rescaled-slopes
echo "--JRC rescaled.--"
tjrc=`date +%s`

#Create country raster
tmfpython3 -m methods.inputs.generate_country_raster --jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--matching /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matching-area.geojson \
--countries /maps/4C/osm_boundaries.geojson \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"countries.tif
echo "--Country raster created.--"
tcountryraster=`date +%s`

#Matching: calculate set K
tmfpython3 -m methods.matching.calculate_k \
--project /maps/epr26/tmf-data/projects/"$proj".geojson \
--start_year "$t0" \
--evaluation_year 2021 \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation /maps/epr26/tmf_pipe_out/"$proj"/rescaled-elevation \
--slope /maps/epr26/tmf_pipe_out/"$proj"/rescaled-slopes \
--access /maps/4C/access \
--countries-raster /maps/epr26/tmf_pipe_out/"$proj"/"$proj"countries.tif \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"k.parquet
echo "--Set K created.--"
tk=`date +%s`

#Matching: calculate set M
tmfpython3 -m methods.matching.find_potential_matches \
--k /maps/epr26/tmf_pipe_out/"$proj"/"$proj"k.parquet \
--matching /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matching-area.geojson \
--start_year "$t0" \
--evaluation_year 2021 \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation /maps/epr26/tmf_pipe_out/"$proj"/rescaled-elevation \
--slope /maps/epr26/tmf_pipe_out/"$proj"/rescaled-slopes \
--access /maps/4C/access \
--countries-raster /maps/epr26/tmf_pipe_out/"$proj"/"$proj"countries.tif \
--output /maps/epr26/tmf_pipe_out/"$proj"/matches
tmfpython3 -m methods.matching.build_m_raster \
--rasters_directory /maps/epr26/tmf_pipe_out/"$proj"/matches \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matches.tif \
-j 20
tmfpython3 -m methods.matching.build_m_table \
--raster /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matches.tif \
--matching /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matching-area.geojson \
--start_year "$t0" \
--evaluation_year 2021 \
--jrc /maps/forecol/data/JRC/v1_2022/AnnualChange/tifs \
--cpc /maps/rhm31/fine_circular_coverage/forecol_complete/ \
--ecoregions /maps/4C/ecoregions/ \
--elevation /maps/epr26/tmf_pipe_out/"$proj"/rescaled-elevation \
--slope /maps/epr26/tmf_pipe_out/"$proj"/rescaled-slopes \
--access /maps/4C/access \
--countries-raster /maps/epr26/tmf_pipe_out/"$proj"/"$proj"countries.tif \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matches.parquet
echo "--Set M created.--"
tm=`date +%s`

#Matching: find pairs
tmfpython3 -m methods.matching.find_pairs \
--k /maps/epr26/tmf_pipe_out/"$proj"/"$proj"k.parquet \
--m /maps/epr26/tmf_pipe_out/"$proj"/"$proj"matches.parquet \
--start_year "$t0" \
--output /maps/epr26/tmf_pipe_out/"$proj"/pairs \
--seed 42 \
-j 1
echo "--Pairs matched.--"
tpairs=`date +%s`

#Calculate additionality
tmfpython3 -m methods.outputs.calculate_additionality \
--project /maps/epr26/tmf-data/projects/"$proj".geojson \
--project_start "$t0" \
--evaluation_year 2021 \
--density /maps/epr26/tmf_pipe_out/"$proj"/"$proj"carbon-density.csv \
--matches /maps/epr26/tmf_pipe_out/"$proj"/pairs \
--output /maps/epr26/tmf_pipe_out/"$proj"/"$proj"additionality.csv
echo "--Additionality calculated.--"
tadditionality=`date +%s`

#end clock
end=`date +%s`
echo "$proj was done" | tee -a out_"$proj"_"$d".txt
echo "t0 was $t0" | tee -a out_"$proj"_"$d".txt
echo Execution time was `expr $end - $start` seconds. | tee -a out_"$proj"_"$d".txt
echo Buffer: `expr $tbuffer - $start` seconds. | tee -a out_"$proj"_"$d".txt
echo Leakage: `expr $tleakage - $tbuffer` seconds. | tee -a out_"$proj"_"$d".txt
echo LUC: `expr $tluc - $tleakage` seconds. | tee -a out_"$proj"_"$d".txt
echo AGB: `expr $tagb - $tluc` seconds. | tee -a out_"$proj"_"$d".txt
echo Country list: `expr $tcountrylist - $tagb` seconds. | tee -a out_"$proj"_"$d".txt
echo Matching area: `expr $tmatcharea - $tcountrylist` seconds. | tee -a out_"$proj"_"$d".txt
echo SRTM: `expr $tsrtm - $tmatcharea` seconds. | tee -a out_"$proj"_"$d".txt
echo Slope: `expr $tslope - $tsrtm` seconds. | tee -a out_"$proj"_"$d".txt
echo JRC tiles: `expr $tjrc - $tslope` seconds. | tee -a out_"$proj"_"$d".txt
echo Country raster: `expr $tcountryraster - $tjrc` seconds. | tee -a out_"$proj"_"$d".txt
echo K set: `expr $tk - $tcountryraster` seconds. | tee -a out_"$proj"_"$d".txt
echo M set: `expr $tm - $tk` seconds. | tee -a out_"$proj"_"$d".txt
echo Pairs: `expr $tpairs - $tm` seconds. | tee -a out_"$proj"_"$d".txt
echo Additionality: `expr $tadditionality - $tpairs` seconds. | tee -a out_"$proj"_"$d".txt