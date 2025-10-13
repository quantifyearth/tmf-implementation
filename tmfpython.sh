#!/bin/bash
# filepath: /home/jh2589/tmf_pipeline/tmf-implementation/python.sh
set -e

CFG="tmfpython.conf"

if [ -f "$CFG" ]; then
  source "$CFG"
else
  echo "ERROR: config file $CFG not found."
  exit 1
fi

SCC_CSV="${SCC_CSV:-/path/to/scc.csv}"

# Define previous settings file in same directory as script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREV_SETTINGS="${SCRIPT_DIR}/python_settings"

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
  [18]="Calculate carbon additionality"
  [19]="Calculate biodiversity additionality"
  [20]="Calculate durability"
)


# Global variables for step selection
STEPS_TO_RUN=()
RUN_ALL=false
STEPS_RAW=""

# Function to save settings for repeat
function save_settings() {
  local mode="$1"
  local steps="$2"
  local proj="$3"
  local t0="$4"
  local eval_year="$5"
  local csv_file="$6"
  
  cat > "$PREV_SETTINGS" << EOF
MODE=$mode
STEPS_RAW=$steps
PROJ=$proj
T0=$t0
EVAL_YEAR=$eval_year
CSV_FILE=$csv_file
EOF
  echo "Settings saved for next run."
}

# Function to load previous settings
function load_previous_settings() {
  if [ -f "$PREV_SETTINGS" ]; then
    source "$PREV_SETTINGS"
    echo "Previous settings loaded:"
    if [ "$MODE" = "single" ]; then
      echo "  Mode: Single project"
      echo "  Project: $PROJ"
      echo "  Start year: $T0"
      echo "  Evaluation year: $EVAL_YEAR"
    elif [ "$MODE" = "csv" ]; then
      echo "  Mode: CSV of projects"
      echo "  CSV file: $CSV_FILE"
    fi
    echo "  Steps: $STEPS_RAW"
    return 0
  else
    echo "No previous settings found."
    return 1
  fi
}

function select_steps {
  STEPS_TO_RUN=()
  RUN_ALL=false
  echo "Which steps would you like to run?"
  echo "  1) All steps"
  echo "  2) Specify steps"
  read -p "Select [1/2]: " MODE_NUM

  # Reset global variable
  STEPS_RAW=""
  
  if [ "$MODE_NUM" = "2" ]; then
    echo "Available steps:"
    keys=( $(printf "%s\n" "${!STEP_DESC[@]}" | sort -n) )
    total=${#keys[@]}
    per_page=20
    for idx in "${!keys[@]}"; do
      i=${keys[idx]}
      printf "  %2d) %s\n" "$i" "${STEP_DESC[$i]}"
      if [ $(((idx+1)%per_page)) -eq 0 ] && [ $((idx+1)) -lt "$total" ]; then
        read -p "-- more -- Press Enter to continue --"
      fi
    done
    read -p "Enter steps (e.g. 3-6,8,10): " STEPS_RAW
    IFS=',' read -ra TOKENS <<< "$STEPS_RAW"
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
    STEPS_RAW="all"
  fi
}

function should_run {
  local S=$1
  $RUN_ALL && return 0
  for x in "${STEPS_TO_RUN[@]}"; do
    [ "$x" -eq "$S" ] && return 0
  done
  return 1
}

function run_single_project {
  read -p "Enter project name (ID): " proj
  read -p "Enter start year: " t0
  read -p "Enter evaluation year: " eval_year
  GEOJSON="${INPUT_DIR}/${proj}.geojson"
  if [ ! -f "$GEOJSON" ]; then
    echo "ERROR: geojson not found for $proj ($GEOJSON)"
    return
  fi
  
  # Save settings immediately after gathering them
  save_settings "single" "$STEPS_RAW" "$proj" "$t0" "$eval_year" ""
  
  echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
  run_pipeline "$proj" "$t0" "$eval_year"
}

function run_csv_projects {
  read -p "Enter CSV file to use [project_metadata.csv]: " CSV_FILE
  CSV_FILE=${CSV_FILE:-project_metadata.csv}
  if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file $CSV_FILE not found."
    return
  fi
  
  # Save settings immediately
  save_settings "csv" "$STEPS_RAW" "" "" "" "$CSV_FILE"
  
  tail -n +2 "$CSV_FILE" | while IFS=',' read -r proj t0 eval_year; do
    GEOJSON="${INPUT_DIR}/${proj}.geojson"
    if [ ! -f "$GEOJSON" ]; then
      echo "Skipping $proj: geojson not found ($GEOJSON)"
      continue
    fi
    echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
    run_pipeline "$proj" "$t0" "$eval_year"
  done
}

function run_pipeline {
  local proj="$1"
  local t0="$2"
  local eval_year="$3"
  local GEOJSON="${INPUT_DIR}/${proj}.geojson"

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
  --k_directory "${OUTPUT_DIR}/${proj}/k_grids" \
  --m_parquet_filename "${OUTPUT_DIR}/${proj}/matches.parquet" \
  --start_year "$t0" \
  --evaluation_year "$eval_year" \
  --carbon_density "${OUTPUT_DIR}/${proj}/carbon-density.csv" \
  --project_area "${INPUT_DIR}/${proj}.geojson" \
  --output_folder "${OUTPUT_DIR}/${proj}/pairs" \
  --seed 42 \
  --batch_size 16 \
  --rse_threshold 0.1 \
  --max_potential_matches 10000 \
  --processes_count 16
fi

if should_run 18; then
  tmfpython3 -m methods.outputs.calculate_additionality \
    --project $GEOJSON \
    --project_start $t0 \
    --evaluation_year $eval_year \
    --density "${OUTPUT_DIR}/${proj}/carbon-density.csv" \
    --matches "${OUTPUT_DIR}/${proj}/pairs" \
    --output "${OUTPUT_DIR}/${proj}/additionality.csv" \
    --grid_output_dir "${OUTPUT_DIR}/${proj}/grid_results" < /dev/null
  echo "--Carbon additionality calculated.--"
fi

if should_run 19; then
  tmfpython3 -m methods.outputs.life_additionality_yearly \
    --life   "${BIODIVERSITY_RASTER}" \
    --project "${INPUT_DIR}/${proj}.geojson" \
    --output_clip  "${OUTPUT_DIR}/${proj}/biodiversity_clip.geojson" \
    --parquet_folder "${OUTPUT_DIR}/${proj}/pairs" \
    --output_folder "${OUTPUT_DIR}/${proj}/life_results" \
    --output_csv    "${OUTPUT_DIR}/${proj}/biodiversity_summary.csv" \
    --min_points    "${MIN_POINTS}"
  echo "--Biodiversity additionality calculated.--"
fi

PROJECT_END_DATE=$((t0 + 40))

if should_run 20; then
  tmfpython3 -m methods.outputs.durability \
    --additionality-csv "${OUTPUT_DIR}/${proj}/additionality.csv" \
    --grid-folder "${OUTPUT_DIR}/${proj}/grid_results/additionality" \
    --scc-csv "${SCC_CSV}" \
    --project-end-date "${PROJECT_END_DATE}" \
    --output-dir "${OUTPUT_DIR}/${proj}" \
    --additionality-percentile 0.2 \
    --control-percentile 0.2 \
    --verbose
  echo "--Durability calculated.--"
fi
}

# Main menu loop
while true; do
  echo ""
  echo "Select mode:"
  echo "  1) Run single project"
  echo "  2) Run CSV of projects"
  if [ -f "$PREV_SETTINGS" ]; then
    echo "  3) Repeat prior instructions"
  fi
  echo "  4) Exit"
  read -p "Enter choice [1-4]: " CHOICE

  case "$CHOICE" in
    1)
      select_steps
      run_single_project
      ;;
    2)
      select_steps
      run_csv_projects
      ;;
    3)
      if [ -f "$PREV_SETTINGS" ]; then
        if load_previous_settings; then
          # Parse stored settings to rebuild STEPS_TO_RUN array
          if [ "$STEPS_RAW" != "" ] && [ "$STEPS_RAW" != "all" ]; then
            STEPS_TO_RUN=()
            IFS=',' read -ra TOKENS <<< "$STEPS_RAW"
            for t in "${TOKENS[@]}"; do
              if [[ $t =~ ^([0-9]+)-([0-9]+)$ ]]; then
                start=${BASH_REMATCH[1]}
                end=${BASH_REMATCH[2]}
                for ((n=start; n<=end; n++)); do
                  STEPS_TO_RUN+=("$n")
                done
              elif [[ $t =~ ^[0-9]+$ ]]; then
                STEPS_TO_RUN+=("$t")
              fi
            done
            RUN_ALL=false
          else
            RUN_ALL=true
          fi

          if [ "$MODE" = "single" ]; then
            GEOJSON="${INPUT_DIR}/${PROJ}.geojson"
            if [ ! -f "$GEOJSON" ]; then
              echo "ERROR: geojson not found for $PROJ ($GEOJSON)"
              continue
            fi
            echo "Running pipeline for $PROJ (start: $T0, eval: $EVAL_YEAR)"
            run_pipeline "$PROJ" "$T0" "$EVAL_YEAR"
          elif [ "$MODE" = "csv" ]; then
            if [ ! -f "$CSV_FILE" ]; then
              echo "ERROR: CSV file $CSV_FILE not found."
              continue
            fi
            tail -n +2 "$CSV_FILE" | while IFS=',' read -r proj t0 eval_year; do
              GEOJSON="${INPUT_DIR}/${proj}.geojson"
              if [ ! -f "$GEOJSON" ]; then
                echo "Skipping $proj: geojson not found ($GEOJSON)"
                continue
              fi
              echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
              run_pipeline "$proj" "$t0" "$eval_year"
            done
          fi
        fi
      else
        echo "No prior instructions to repeat."
      fi
      ;;
    4)
      echo "Exiting."
      exit 0
      ;;
    *)
      echo "Invalid choice."
      ;;
  esac
done