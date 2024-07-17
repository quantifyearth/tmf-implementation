# Running the pipeline

This documents a simple run through of the pipeline for a single project. It assumes you are starting from scratch - most times you don't need to run the full pipeline as you already have the inputs, and just run the matching sections.

Assumptions:

* You have set up your python environment as per the top level README.
* You are executing these commands from the root directory of this repository.
* You have a target directory that can cope with all the many gigabytes of data required.
* You have a POSTGIS database set up for holding the GEDI data.

To make it easier to follow, I will assume you are keeping your data generally in `/data/tmf`

# Build environment

```shark-build:gdalenv
((from ghcr.io/osgeo/gdal:ubuntu-small-3.6.4)
 (run (network host) (shell "apt-get update -qqy && apt-get -y install python3-pip libpq-dev git && rm -rf /var/lib/apt/lists/* && rm -rf /var/cache/apt/*"))
 (run (network host) (shell "pip install --upgrade pip"))
 (run (network host) (shell "pip install numpy"))
 (run (network host) (shell "pip install gdal[numpy]==3.6.4"))
 (workdir "/usr/src/app")
 (copy (src "requirements.txt") (dst "./"))
 (run (network host) (shell "pip install --no-cache-dir -r requirements.txt"))
 (copy (src ".") (dst "./"))
 (run (shell "make lint && make type && make test"))
)
```


# Downloading resources used by all projects

First we need to download the layers used by all projects, and in some cases convert them into a more useful form.

## JRC

JRC data is generally downloaded first, as all other tiles are at JRC resolution, and so often things that don't need JRC data will still read one JRC tile just to get the correct GeoTIFF projection and pixel scale. To fetch JRC data we do:

```shark-run:gdalenv
python3 -m methods.inputs.download_jrc_data /data/tmf/jrc/zips /data/tmf/jrc/tif
```

The zips are kept around for archival purposes, due to known difficultly in versioning JRC data, but only the tifs directory is needed from here on.

We also want to generate the Finegrain Circular CPC (FCC) maps. This stage is very expensive, taking days to run, but only needs to be done once per year when JRC updates.

```shark-run:gdalenv
python3 -m methods.inputs.generate_fine_circular_coverage --jrc /data/tmf/jrc/tif --output /data/tmf/fcc-cpcs
```


## Ecoregions

First we download the ecoregions as a shape file:

```shark-run:gdalenv
python3 -m methods.inputs.download_shapefiles ecoregion /data/tmf/ecoregions/ecoregions.geojson
```

Then we convert it into raster files. This takes up more space, but the ecoregions are slow to raster on demand each time, so this is faster doing it just once.

```shark-run:gdalenv
python3 -m methods.inputs.generate_ecoregion_rasters /data/tmf/ecoregions/ecoregions.geojson /data/tmf/jrc/tif /data/tmf/ecoregions
```

## ACCESS

Firstly we have access to healthcare data:

```shark-run:gdalenv
python3 -m methods.inputs.download_accessibility /data/tmf/access/raw.tif
```

Which must then be turned into smaller tiles:

```shark-run:gdalenv
python3 -m methods.inputs.generate_access_tiles /data/tmf/access/raw.tif /data/tmf/jrc/tif /data/tmf/access
```

## Country boarders

We use OSM data for country borders data. This requires an API key to access the downloads. To get an API key you must follow these steps:

1. Sign up to [openstreetmap.org](https://openstreetmaps.com)
2. Use that account to sign into [osm-boundaries.com](https://osm-boundaries.com/)
3. In the top right of osm-boundaries, click on your profuile
4. Copy your API key from here, and replace the XXXXXXXX below with that.

```shark-run:gdalenv
export OSM_BOUNDARIES_KEY="XXXXXXXX"
python3 -m methods.inputs.download_osm_countries /data/tmf/osm_borders.geojson
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

```shark-run:gdalenv
python3 -m methods.inputs.generate_boundary --project /data/tmf/project_boundaries/123.geojson --output /data/tmf/123/buffer.geojson
```

We also want a shape for the leakage zone:

```shark-run:gdalenv
python3 -m methods.inputs.generate_leakage --project /data/tmf/project_boundaries/123.geojson --output /data/tmf/123/leakage.geojson
```

## GEDI data

The GEDI data is used to calculate the AGB per LUC for a project. This data is downloaded from the NASA Earthdata portal, and to do that this project uses the [biomassrecovery](https://github.com/ameliaholcomb/biomass-recovery) library, and that assumes you have a `.env` file in your home directory that looks like this:

```
# Earthdata access
EARTHDATA_USER="XXXXXXXX"
EARTHDATA_PASSWORD="XXXXXXXX"
EARTH_DATA_COOKIE_FILE="/home/myusername/.urs_cookies"

# User path constants
USER_PATH="/home/myusername"
DATA_PATH="/data/tmf/gedi"
```

Once you have this the download script to pull the raw GEDI data is:

```shark-run:gdalenv
python3 -m methods.inputs.locate_gedi_data /data/tmf/123/buffer.geojson /data/tmf/gedi/granule/info/
python3 -m methods.inputs.download_gedi_data /data/tmf/gedi/granule/info/* /data/tmf/gedi/granule/
```

Each GEDI granule is a complete sweep from the ISS along the path that crossed the buffer zone, and so contains
much more data than we need. We thus need to filter this to give us just the GEDI shots we're interested in. At the
same time we filter based on time and quality as per the methodology document:

```shark-run:gdalenv
python3 -m methods.inputs.filter_gedi_data --buffer /data/tmf/123/buffer.geojson \
                                          --granules /data/tmf/gedi/granule/ \
                                          --output /data/tmf/123/gedi.geojson
```

Once filtered we can then combine the GEDI data with the JRC data we have to work out the carbon density per land usage class:

```shark-run:gdalenv
python3 -m methods.inputs.generate_carbon_density --jrc /data/tmf/jrc/tif \
                                                  --gedi /data/tmf/123/gedi.geojson \
                                                  --output /data/tmf/123/carbon-density.csv
```

## Generate matching area

We need a list of countries that a project intersects with to calculate the matching area:

```shark-run:gdalenv
python3 -m methods.inputs.generate_country_list --leakage /data/tmf/project_boundaries/123.geojson \
                                               --countries /data/tmf/osm_borders.geojson \
                                               --output /data/tmf/123/country-list.json
```

Note that this stage requires a full list of other project boundaries that must be avoided for matching purposes, which is in `/data/tmf/project_boundaries`


```shark-run:gdalenv
python3 -m methods.inputs.generate_matching_area --project /data/tmf/project_boundaries/123.geojson \
                                                  --countrycodes /data/tmf/123/country-list.json \
                                                  --countries /data/tmf/osm_borders.geojson \
                                                  --ecoregions /data/tmf/ecoregions/ecoregions.geojson \
                                                  --projects /data/tmf/project_boundaries \
                                                  --output /data/tmf/project_boundaries/123/matching-area.geojson
```

## Elevation and slope data

We use the SRTM elevation data, which we download to cover the matching area:

```shark-run:gdalenv
python3 -m methods.inputs.download_srtm_data --project /data/tmf/project_boundaries/123.geojson \
                                            --matching /data/tmf/project_boundaries/123/matching-area.geojson \
                                            --zips /data/tmf/srtm/zip \
                                            --tifs /data/tmf/srtm/tif
```

Then from that generate slope data using GDAL:

```shark-run:gdalenv
python3 -m methods.inputs.generate_slope --input /data/tmf/srtm/tif --output /data/tmf/slopes
```

Once we have that we need to rescale the data to match JRC resolution:

```shark-run:gdalenv
python3 -m methods.inputs.rescale_tiles_to_jrc --jrc /data/tmf/jrc/tif \
                                                 --tiles /data/tmf/srtm/tif \
                                                 --output /data/tmf/rescaled-elevation
python3 -m methods.inputs.rescale_tiles_to_jrc --jrc /data/tmf/jrc/tif \
                                                 --tiles /data/tmf/slopes \
                                                 --output /data/tmf/rescaled-slopes
```

## Country raster

Again, rather than repeatedly dynamically rasterize the country vectors, we rasterise them once and re-use them:

```shark-run:gdalenv
python3 -m methods.inputs.generate_country_raster --jrc /data/tmf/jrc/tif \
                                                  --matching /data/tmf/project_boundaries/123/matching-area.geojson \
                                                  --countries /data/tmf/osm_borders.geojson \
                                                  --output /data/tmf/123/countries.tif
```

# Pixel matching

Pixel matching is split into three main stages: Calculating K, then M, and then finding pairs between them.

## Calculate set K

First we make K.

```shark-run:gdalenv
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

```shark-run:gdalenv
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

```shark-run:gdalenv
python3 -m methods.matching.build_m_raster --rasters_directory /data/tmf/123/matches \
                                          --output /data/tmf/123/matches.tif \
                                          -j 20
```

We then convert that raster into a table of pixels plus the data we need:

```shark-run:gdalenv
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

```shark-run:gdalenv
python3 -m methods.matching.find_pairs --k /data/tmf/123/k.parquet \
                                         --m /data/tmf/123/matches.parquet \
                                         --start_year 2012 \
                                         --output /data/tmf/123/pairs \
                                         --seed 42 \
                                         -j 1
```

# Calculate additionality

Finally this script calculates additionality:

```shark-run:gdalenv
python3 -m methods.outputs.calculate_additionality --project /data/tmf/project_boundaries/123.geojson \
                                                    --project_start 2012 \
                                                    --evaluation_year 2021 \
                                                    --density /data/tmf/123/carbon-density.csv \
                                                    --matches /data/tmf/123/pairs \
                                                    --output /data/tmf/123/additionality.csv
```

By running the additionality step with the environment variable `TMF_PARTIALS` set to some directory, this step will also generate GeoJSON files for visualising the pairs and the standardised mean difference calculations for the matching variables. You can add `TMF_PARTIALS=/some/dir` before `python3` to set this just for a specific run of `calculate_additionality`. 
