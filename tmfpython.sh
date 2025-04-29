#!/bin/bash
# filepath: /home/jh2589/tmf_pipeline/tmf-implementation/tmfpython.sh
set -e

####### 1) Load user config (one line per VAR=VALUE) #######
CFG="tmfpython.conf"
if [ -f "$CFG" ]; then
  source "$CFG"
else
  echo "ERROR: config file $CFG not found. Create it with:"
  echo "  INPUT_DIR=…"
  echo "  OUTPUT_DIR=…"
  echo "  JRC_DIR=…"
  echo "  FCC_DIR=…"
  echo "  ECOR_DIR=…"
  echo "  ECOR_GEOJSON=…"
  echo "  ELEV_DIR=…"
  echo "  SLOPE_DIR=…"
  echo "  ACCESS_DIR=…"
  echo "  COUNTRIES_RASTER=…"
  echo "  OTHER_PROJECTS_DIR=…"
  echo "  GEDI_INFO_DIR=…"
  echo "  GEDI_DATA_DIR=…"
  echo "  SRTM_ZIP_DIR=…"
  echo "  SRTM_TIF_DIR=…"
  exit 1
fi

####### 2) Define step descriptions #######
declare -A STEP_DESC=(
  [1]="Create output folder"
  [2]="Generate buffer (boundary)"
  [3]="Locate GEDI data"
  [4]="Download GEDI data"
  [5]="Filter GEDI data"
  [6]="Generate carbon density"
  [7]="Generate country list"
  [8]="Generate matching area"
  [9]="Download SRTM data"
  [10]="Generate slopes"
  [11]="Rescale elevation tiles"
  [12]="Rescale slope tiles"
  [13]="Generate country raster"
  [14]="Calculate set K"
  [15]="Find potential matches"
  [16]="Build M table"
  [17]="Find pairs"
  [18]="Calculate additionality"
)

echo "Which steps would you like to run?"
echo "  1) All steps"
echo "  2) Specify steps"
read -p "Select [1/2]: " MODE

declare -a STEPS_TO_RUN=()
RUN_ALL=false

if [ "$MODE" = "2" ]; then
  echo "Available steps:"
  # collect keys in numeric order
  keys=( $(printf "%s\n" "${!STEP_DESC[@]}" | sort -n) )
  total=${#keys[@]}
  per_page=20

  for idx in "${!keys[@]}"; do
    i=${keys[idx]}
    printf "  %2d) %s\n" "$i" "${STEP_DESC[$i]}"

    # after every $per_page items (but not after the last), pause
    if [ $(((idx+1)%per_page)) -eq 0 ] && [ $((idx+1)) -lt "$total" ]; then
      read -p "-- more -- Press Enter to continue --"
    fi
  done

  read -p "Enter steps (e.g. 3-6,8,10): " RAW
  # parse RAW into STEPS_TO_RUN
  IFS=',' read -ra TOKENS <<< "$RAW"
  for t in "${TOKENS[@]}"; do
    if [[ $t =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      for ((n=start; n<=end; n++)); do
        STEPS_TO_RUN+=("$n")
      done
    elif [[ $t =~ ^[0-9]+$ ]]; then
      STEPS_TO_RUN+=("$t")
    else
      echo "Warning: invalid token '$t' skipped"
    fi
  done
  RUN_ALL=false
else
  RUN_ALL=true
fi

read -p "Project name (no .geojson): " proj
read -p "Start year (t0): " t0
read -p "Evaluation year: " eval_year

function should_run {
  local S=$1
  $RUN_ALL && return 0
  for x in "${STEPS_TO_RUN[@]}"; do
    [ "$x" -eq "$S" ] && return 0
  done
  return 1
}

####### 3) Steps #######
####### 3) Steps #######
if should_run 1; then
  mkdir -p "${OUTPUT_DIR}/${proj}"
  echo "--Folder created.--"
fi

if should_run 2; then
  tmfpython3 -m methods.inputs.generate_boundary \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --output "${OUTPUT_DIR}/${proj}/buffer.geojson"
  echo "--Buffer created.--"
fi

if should_run 3; then
  tmfpython3 -m methods.inputs.locate_gedi_data \
    --granules "${GEDI_INFO_DIR}" \
    --buffer "${OUTPUT_DIR}/${proj}/buffer.geojson" \
    --output "${OUTPUT_DIR}/${proj}/gedi_names.csv"
  echo "--GEDI names located.--"
fi

if should_run 4; then
  tmfpython3 -m methods.inputs.download_gedi_data \
    --granules "${GEDI_INFO_DIR}" \
    --output   "${GEDI_DATA_DIR}"
  echo "--GEDI data downloaded.--"
fi

if should_run 5; then
  tmfpython3 -m methods.inputs.filter_gedi_data \
    --granules "${GEDI_DATA_DIR}" \
    --buffer "${OUTPUT_DIR}/${proj}/buffer.geojson" \
    --csv "${OUTPUT_DIR}/${proj}/gedi_names.csv" \
    --output "${OUTPUT_DIR}/${proj}/gedi.geojson"
  echo "--GEDI filtered.--"
fi

if should_run 6; then
  tmfpython3 -m methods.inputs.generate_carbon_density \
    --jrc "${JRC_DIR}" \
    --gedi "${OUTPUT_DIR}/${proj}/gedi.geojson" \
    --output "${OUTPUT_DIR}/${proj}/carbon-density.csv"
  echo "--Carbon density created.--"
fi

if should_run 7; then
  tmfpython3 -m methods.inputs.generate_country_list \
    --buffer "${OUTPUT_DIR}/${proj}/buffer.geojson" \
    --countries "${COUNTRIES_RASTER}" \
    --output "${OUTPUT_DIR}/${proj}/country-list.json"
  echo "--Country list created.--"
fi

if should_run 8; then
  tmfpython3 -m methods.inputs.generate_matching_area \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --countrycodes "${OUTPUT_DIR}/${proj}/country-list.json" \
    --countries "${COUNTRIES_RASTER}" \
    --ecoregions "${ECOR_GEOJSON}" \
    --projects "${OTHER_PROJECTS_DIR}" \
    --output "${OUTPUT_DIR}/${proj}/matching-area.geojson"
  echo "--Matching area created.--"
fi

if should_run 9; then
  tmfpython3 -m methods.inputs.download_srtm_data \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --matching "${OUTPUT_DIR}/${proj}/matching-area.geojson" \
    --zips "${SRTM_ZIP_DIR}" \
    --tifs "${SRTM_TIF_DIR}"
  echo "--SRTM downloaded.--"
fi

if should_run 10; then
  tmfpython3 -m methods.inputs.generate_slope \
    --input "${SRTM_TIF_DIR}" \
    --output "${OUTPUT_DIR}/slopes"
  echo "--Slope created.--"
fi

if should_run 11; then
  tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
    --jrc "${JRC_DIR}" \
    --tiles "${SRTM_TIF_DIR}" \
    --output "${OUTPUT_DIR}/rescaled-elevation"
  echo "--Elevation rescaled.--"
fi

if should_run 12; then
  tmfpython3 -m methods.inputs.rescale_tiles_to_jrc \
    --jrc "${JRC_DIR}" \
    --tiles "${OUTPUT_DIR}/slopes" \
    --output "${OUTPUT_DIR}/rescaled-slopes"
  echo "--Slopes rescaled.--"
fi

if should_run 13; then
  tmfpython3 -m methods.inputs.generate_country_raster \
    --jrc "${JRC_DIR}" \
    --matching "${OUTPUT_DIR}/${proj}/matching-area.geojson" \
    --countries "${COUNTRIES_RASTER}" \
    --output "${OUTPUT_DIR}/${proj}/countries.tif"
  echo "--Country raster created.--"
fi

if should_run 14; then
  tmfpython3 -m methods.matching.calculate_k \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --start_year "$t0" \
    --evaluation_year "$eval_year" \
    --jrc "${JRC_DIR}" \
    --fcc "${FCC_DIR}" \
    --ecoregions "${ECOR_DIR}" \
    --elevation "${OUTPUT_DIR}/rescaled-elevation" \
    --slope "${OUTPUT_DIR}/rescaled-slopes" \
    --access "${ACCESS_DIR}" \
    --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
    --output "${OUTPUT_DIR}/${proj}/k_grids"
  echo "--Set K created.--"
fi

if should_run 15; then
  tmfpython3 -m methods.matching.find_potential_matches \
    --k "${OUTPUT_DIR}/${proj}/k_grids" \
    --matching "${OUTPUT_DIR}/${proj}/matching-area.geojson" \
    --start_year "$t0" \
    --evaluation_year "$eval_year" \
    --jrc "${JRC_DIR}" \
    --fcc "${FCC_DIR}" \
    --ecoregions "${ECOR_DIR}" \
    --elevation "${OUTPUT_DIR}/rescaled-elevation" \
    --slope "${OUTPUT_DIR}/rescaled-slopes" \
    --access "${ACCESS_DIR}" \
    --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
    --output "${OUTPUT_DIR}/${proj}/matches"
  echo "--M rasters created.--"
fi

if should_run 16; then
  tmfpython3 -m methods.matching.build_m_table \
    --rasters_directory "${OUTPUT_DIR}/${proj}/matches" \
    --matching "${OUTPUT_DIR}/${proj}/matching-area.geojson" \
    --start_year "$t0" \
    --evaluation_year "$eval_year" \
    --jrc "${JRC_DIR}" \
    --fcc "${FCC_DIR}" \
    --ecoregions "${ECOR_DIR}" \
    --elevation "${OUTPUT_DIR}/rescaled-elevation" \
    --slope "${OUTPUT_DIR}/rescaled-slopes" \
    --access "${ACCESS_DIR}" \
    --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
    --output "${OUTPUT_DIR}/${proj}/matches.parquet"
  echo "--Set M created.--"
fi

if should_run 17; then
  tmfpython3 -m methods.matching.find_pairs \
    --k "${OUTPUT_DIR}/${proj}/k_grids" \
    --m "${OUTPUT_DIR}/${proj}/matches.parquet" \
    --start_year "$t0" \
    --evaluation_year "$eval_year" \
    --density "${OUTPUT_DIR}/${proj}/carbon-density.csv" \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --output "${OUTPUT_DIR}/${proj}/pairs" \
    --seed 42 \
    --batch_size 10 \
    --rse_threshold 0.025 \
    --j $(nproc)
  echo "--Pairs matched.--"
fi

if should_run 18; then
  tmfpython3 -m methods.outputs.calculate_additionality \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --project_start "$t0" \
    --evaluation_year "$eval_year" \
    --density "${OUTPUT_DIR}/${proj}/carbon-density.csv" \
    --matches "${OUTPUT_DIR}/${proj}/pairs" \
    --output "${OUTPUT_DIR}/${proj}/additionality.csv" \
    --partials "${OUTPUT_DIR}/${proj}/partials"
  echo "--Additionality calculated.--"
fi