---
inputs:
- /data/tmf/project_boundaries/123.geojson
- /data/tmf/project_boundaries
---
# Running the pipeline

This documents a simple run through of the pipeline for a single project. It assumes you are starting from scratch - most times you don't need to run the full pipeline as you already have the inputs, and just run the matching sections.

Assumptions:

* You have set up your python environment as per the top level README.
* You are executing these commands from the root directory of this repository.
* You have a target directory that can cope with all the many gigabytes of data required.
* You have a POSTGIS database set up for holding the GEDI data.

To make it easier to follow, I will assume you are keeping your data generally in `/data/tmf`

# Downloading resources used by all projects

First we need to download the layers used by all projects, and in some cases convert them into a more useful form.

## JRC

JRC data is generally downloaded first, as all other tiles are at JRC resolution, and so often things that don't need JRC data will still read one JRC tile just to get the correct GeoTIFF projection and pixel scale. To fetch JRC data we do:

```ShellSession
python3 -m methods.inputs.download_jrc_data /data/tmf/jrc/zips /data/tmf/jrc/tif
```

The zips are kept around for archival purposes, due to known difficultly in versioning JRC data, but only the tifs directory is needed from here on.

We also want to generate the Finegrain Circular CPC (FCC) maps. This stage is very expensive, taking days to run, but only needs to be done once per year when JRC updates.

```ShellSession
python3 -m methods.inputs.generate_fine_circular_coverage --jrc /data/tmf/jrc/tif --output /data/tmf/fcc-cpcs
```


## Ecoregions

First we download the ecoregions as a shape file:

```ShellSession
python3 -m methods.inputs.download_shapefiles ecoregion /data/tmf/ecoregions/ecoregions.geojson
```

Then we convert it into raster files. This takes up more space, but the ecoregions are slow to raster on demand each time, so this is faster doing it just once.

```ShellSession
python3 -m methods.inputs.generate_ecoregion_rasters /data/tmf/ecoregions/ecoregions.geojson /data/tmf/jrc/tif /data/tmf/ecoregions
```

## ACCESS

Firstly we have access to healthcare data:

```ShellSession
python3 -m methods.inputs.download_accessibility /data/tmf/access/raw.tif
```

Which must then be turned into smaller tiles:

```ShellSession
python3 -m methods.inputs.generate_access_tiles /data/tmf/access/raw.tif /data/tmf/jrc/tif /data/tmf/access
```

## Country boarders

We use OSM data for country board data:

```ShellSession
python3 -m methods.inputs.osm_countries /data/tmf/osm_borders.geojson
```

This file file is heavy to work with as geojson, so it'll later be rendered to a GeoTIFF.

# Download resources for a specific project

For this section you will now need to have the following:

* A GEOJSON of the project boundaries. For this documentation we will assume you're assessing project `123` and its boundary file is at `/data/tmf/project_boundaries/123.geojson`
* A list of other projects to avoid. For simplicity we assume in this documentation that you have all your project boundaries in `/data/tmf/projects`.
* The start year of the project. For this we use the year 2012.
* A directory to store the results in. For this document we put the results in `/data/tmf/123`

## Make variations on project shapes

We add a 30km buffer around the project for generating land usage and AGB data:

```ShellSession
python3 -m methods.inputs.generate_boundary --project /data/tmf/project_boundaries/123.geojson --output /data/tmf/123/buffer.geojson
```

We also want a shape for the leakage zone:

```ShellSession
python3 -m methods.inputs.generate_leakage --project /data/tmf/project_boundaries/123.geojson --output /data/tmf/123/leakage.geojson
```

## Make LUC tiles

We conver the JRC tiles binary tiles per LUC. In theory we could do this for all JRC tiles, but to save space we just calculate the areas we need.

```ShellSession
python3 -m methods.inputs.generate_luc_layer --buffer /data/tmf/123/buffer.geojson \
                                            --jrc /data/tmf/jrc/tif \
                                            --output /data/tmf/123/luc.tif
```

NB: In theory we could remove this stage if we updated `generate_carbon_density.py` to use the GroupLayers of yirgacheffe that we added later on and is  used by other parts of the pipeline.

## GEDI data

The GEDI data is used to calculate the AGB per LUC for a project. This data is downloaded first, then inserted into a POSTGIS database, and then summarised from that. This project uses the [biomassrecovery](https://github.com/ameliaholcomb/biomass-recovery), and that assumes you have a `.env` file in your home directory that looks like this:

```
# Earthdata access
EARTHDATA_USER="XXXXXXXX"
EARTHDATA_PASSWORD="XXXXXXXX"
EARTH_DATA_COOKIE_FILE="/home/myusername/.urs_cookies"

# User path constants
USER_PATH="/home/myusername"
DATA_PATH="/data/tmf/gedi"

# Database constants
DB_HOST="mypostgreshost"
DB_NAME="tmf_gedi"
DB_USER="tmf_database_user"
DB_PASSWORD="XXXXXXXX"
```

Once you have this the download script is:

```ShellSession
python3 -m methods.inputs.download_gedi_data /data/tmf/123/buffer.geojson /data/tmf/gedi
```

Then the following script does the actual ingest into POSTGIS:

```ShellSession
python3 -m methods.inputs.import_gedi_data /data/tmf/gedi
```

Once the data has been downloaded and ingested into POSTGIS, you then need to generate AGB data layers:

```ShellSession
python3 -m methods.inputs.generate_carbon_density --buffer /data/tmf/123/buffer.geojson \
                                                --luc /data/tmf/123/luc.tif \
                                                --output /data/tmf/123/carbon-density.csv
```

This is the only part of the pipeline that needs the PostGIS server access.

## Generate matching area

We need a list of countries that a project intersects with to calculate the matching area:

```ShellSession
python3 -m methods.inputs.generate_country_list --leakage /data/tmf/project_boundaries/123.geojson \
                                               --countries /data/tmf/osm_borders.geojson \
                                               --output /data/tmf/123/country-list.json
```

Note that this stage requires a full list of other project boundaries that must be avoided for matching purposes, which is in `/data/tmf/project_boundaries`


```ShellSession
python3 -m methods.inputs.generate_matching_area --project /data/tmf/project_boundaries/123.geojson \
                                                  --countrycodes /data/tmf/123/country-list.json \
                                                  --countries /data/tmf/osm_borders.geojson \
                                                  --ecoregions /data/tmf/ecoregions/ecoregions.geojson \
                                                  --projects /data/tmf/project_boundaries \
                                                  --output /data/tmf/123/matching-area.geojson
```

## Elevation and slope data

We use the SRTM elevation data, which we download to cover the matching area:

```ShellSession
python3 -m methods.inputs.download_srtm_data --project /data/tmf/project_boundaries/123.geojson \
                                            --matching /data/tmf/123/matching-area.geojson \
                                            --zips /data/tmf/srtm/zip \
                                            --tifs /data/tmf/srtm/tif
```

Then from that generate slope data using GDAL:

```ShellSession
python3 -m methods.inputs.generate_slope --input /data/tmf/srtm/tif --output /data/tmf/slopes
```

Once we have that we need to rescale the data to match JRC resolution:

```ShellSession
python3 -m methods.inputs.rescale_tiles_to_jrc --jrc /data/tmf/jrc/tif \
                                                 --tiles /data/tmf/srtm/tif \
                                                 --output /data/tmf/rescaled-elevation
python3 -m methods.inputs.rescale_tiles_to_jrc --jrc /data/tmf/jrc/tif \
                                                 --tiles /data/tmf/slopes \
                                                 --output /data/tmf/rescaled-slopes
```

## Country raster

Again, rather than repeatedly dynamically rasterize the country vectors, we rasterise them once and re-use them:

```ShellSession
python3 -m methods.inputs.generate_country_raster --jrc /data/tmf/jrc/tif \
                                                  --matching /data/tmf/123/matching-area.geojson \
                                                  --countries /data/tmf/osm_borders.geojson \
                                                  --output /data/tmf/123/countries.tif
```

# Pixel matching

Pixel matching is split into three main stages: Calculating K, then M, and then finding pairs between them.

## Calculate set K

First we make K.

```ShellSession
python3 -m methods.matching.calculate_k --project /data/tmf/project_boundaries/123.geojson \
                                          --start_year 2012 \
                                          --evaluation_year 2021 \
                                          --jrc /data/tmf/jrc/tif \
                                          --cpc /data/tmf/fcc-cpcs \
                                          --ecoregions /data/tmf/ecoregions \
                                          --elevation /data/tmf/rescaled-elevation \
                                          --slope /data/tmf/rescaled-slopes \
                                          --access /data/tmf/access \
                                          --countries-raster /data/tmf/123/countries.tif \
                                          --output /data/tmf/123/k.parquet
```

## Calculate set M

Calculating the set M is a three step process. First we generate a raster per K that has the matches for that particular pixel:

```ShellSession
python3 -m methods.matching.find_potential_matches --k /data/tmf/123/k.parquet \
                                                     --matching /data/tmf/123/matching-area.geojson \
                                                     --start_year 2012 \
                                                     --evaluation_year 2021 \
                                                     --jrc /data/tmf/jrc/tif \
                                                     --cpc /data/tmf/fcc-cpcs \
                                                     --ecoregions /data/tmf/ecoregions \
                                                     --elevation /data/tmf/rescaled-elevation \
                                                     --slope /data/tmf/rescaled-slopes \
                                                     --access /data/tmf/access \
                                                     --countries-raster /data/tmf/123/countries.tif \
                                                     --output /data/tmf/123/matches
```

Then these rasters get combined into a single raster that is all the potential matches as one. The j parameter controls how many concurrent processes the script can use, which is bounded mostly by how much memory you have available. The value 20 is good for our server, but may not match yours.

```ShellSession
python3 -m methods.matching.build_m_raster --rasters_directory /data/tmf/123/matches \
                                          --output /data/tmf/123/matches.tif \
                                          -j 20
```

We then convert that raster into a table of pixels plus the data we need:

```ShellSession
python3 -m methods.matching.build_m_table --raster /data/tmf/123/matches.tif \
                                            --matching /data/tmf/123/matching-area.geojson \
                                            --start_year 2012 \
                                            --evaluation_year 2021 \
                                            --jrc /data/tmf/jrc/tif \
                                            --cpc /data/tmf/fcc-cpcs \
                                            --ecoregions /data/tmf/ecoregions \
                                            --elevation /data/tmf/rescaled-elevation \
                                            --slope /data/tmf/rescaled-slopes \
                                            --access /data/tmf/access \
                                            --countries-raster /data/tmf/123/countries.tif \
                                            --output /data/tmf/123/matches.parquet
```

## Find pairs

Finally we can find the 100 sets of matched pairs. The seed is used to control the random number generator, and using the same seed on each run ensures consistency despite randomness being part of the selection process.

```ShellSession
python3 -m methods.matching.find_pairs --k /data/tmf/123/k.parquet \
                                         --m /data/tmf/123/matches.parquet \
                                         --start_year 2012 \
                                         --output /data/tmf/123/pairs \
                                         --seed 42 \
                                         -j 1
```

# Calculate additionality

Finally this script calculates additionality:

```ShellSession
python3 -m methods.outputs.calculate_additionality --project /data/tmf/project_boundaries/123.geojson \
                                                    --project_start 2012 \
                                                    --evaluation_year 2021 \
                                                    --density /data/tmf/123/carbon-density.csv \
                                                    --matches /data/tmf/123/pairs \
                                                    --output /data/tmf/123/additionality.csv
```

By running the additionality step with the environment variable `TMF_PARTIALS` set to some directory, this step will also generate GeoJSON files for visualising the pairs and the standardised mean difference calculations for the matching variables. You can add `TMF_PARTIALS=/some/dir` before `python3` to set this just for a specific run of `calculate_additionality`. 
