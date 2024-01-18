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

```shell
$ python3 -m methods.inputs.download_jrc_data /data/tmf/jrc/zips /data/tmf/jrc/tif
```

The zips are kept around for archival purposes, due to known difficultly in versioning JRC data, but only the tifs directory is needed from here on.

We also want to generate the FCC maps. This stage is very expensive, but only needs to be done once per year when JRC updates.

```shell
$ python -m methods.inputs.generate_fine_circular_coverage --jrc /data/tmf/jrc/tif/products/tmf_v1/AnnualChange --output /data/tmf/fcc-cpcs
```


## Ecoregions

First we download the ecoregions as a shape file:

```shell
$ python3 -m methods.inputs.download_shapefiles ecoregion /data/tmf/ecoregions/ecoregions.geojson
```

Then we convert it into raster files. This takes up more space, but the ecoregions are slow to raster on demand each time, so this is faster doing it just once.

```shell
$ python3 -m methods.inputs.generate_ecoregion_rasters /data/tmf/ecoregions/ecoregions.geojson /data/tmf/jrc/tif/products/tmf_v1/AnnualChange /data/tmf/ecoregions
```

## ACCESS

Firstly we have access to healthcare data:

```shell
$ python3 -m methods.inputs.download_accessibility /data/tmf/access/raw.tif
```

Which must then be turned into smaller tiles:

```shell
$ python3 -m methods.inputs.generate_access_tiles /data/tmf/access/raw.tif /data/tmf/jrc/tif /data/tmf/access
```

## Country boarders

We use OSM data for country board data:

```shell
$ python3 -m methods.inputs.osm_countries /data/tmf/osm_borders.geojson
```

# Download resources for a specific project

For this section you will now need to have a project JSON file and a GEOJSON of the project boundaries. The JSON file should look like:

```json
{
	"vcs_id": 123,
	"country_code": "SE",
	"project_start": 2012,
	"quality": "high"
}
```

We will assume they are kept in `/data/tmf/projects/123/data.json` and `/data/tmf/projects/123/boundary.geojson` respectively.

## Make variations on project shapes

We add a 30km buffer around the project for generating land usage and AGB data:

```shell
$ python3 -m methods.inputs.generate_boundary --project /data/tmf/projects/123/boundary.geojson --output /data/tmf/123/buffer.geojson
```

We also want a shape for the leakage zone:

```shell
$ python3 -m methods.inputs.generate_leakage --project /data/tmf/projects/123/boundary.geojson --output /data/tmf/123/leakage.geojson
```

## Make LUC tiles

We conver the JRC tiles binary tiles per LUC. In theory we could do this for all JRC tiles, but to save space we just calculate the areas we need.

```shell
$ python -m methods.inputs.generate_luc_layer /data/tmf/123/buffer.geojson /data/tmf/jrc/tif/products/tmf_v1/AnnualChange /data/tmf/123/luc.tif
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

```shell
$ python3 -m methods.inputs.download_gedi_data /data/tmf/123/buffer.geojson /data/tmf/gedi
```

Then the following script does the actual ingest into POSTGIS:

```shell
$ python3 -m methods.inputs.import_gedi_data /data/tmf/gedi
```

Once the data has been downloaded and ingested into POSTGIS, you then need to generate AGB data layers:

```shell
$ python3 -m methods.inputs.generate_carbon_density /data/tmf/123/buffer.geojson /data/tmf/123/luc.tif /data/tmf/123/carbon-density.csv
```


## Generate matching area

We need a list of countries that a project intersects with to calculate the matching area:

```shell
python -m methods.inputs.generate_country_list --leakage /data/tmf/projects/123/boundary.geojson \
                                               --countries /data/tmf/osm_borders.geojson \
                                               --output /data/tmf/projects/123/country-list.json
```

Note that this stage requires a full list of other project boundaries that must be avoided for matching purposes, which is in `/data/tmf/project_boundaries`


```shell
$ python -m methods.inputs.generate_matching_area --project /data/tmf/projects/123/boundary.geojson \
                                                  --countrycodes /data/tmf/projects/123/country-list.json \
                                                  --countries /data/tmf/osm_borders.geojson \
                                                  --ecoregions /data/tmf/ecoregions/ecoregions.geojson \
                                                  --projects /data/tmf/project_boundaries \
                                                  --output /data/tmf/projects/123/matching-area.geojson
```

## Elevation and slope data



# Pixel matching

## Calculate K

```shell
$ python3 -m methods.matching.calculate_k --project /data/tmf/projects/123/boundary.geojson \
                                          --start_year 2012 \
                                          --evaluation_year 2021 \
                                          --jrc /data/tmf/jrc/tif/products/tmf_v1/AnnualChange \
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

```shell
$ python3 -m methods.matching.find_potential_matches --k /data/tmf/123/k.parquet \
                                                     --matching /data/tmf/123/matching-area.geojson \
                                                     --start_year 2012 \
                                                     --evaluation_year 2021 \
                                                     --jrc /data/tmf/jrc/tif/products/tmf_v1/AnnualChange \
                                                     --cpc /data/tmf/fcc-cpcs \
                                                     --ecoregions /data/tmf/ecoregions \
                                                     --elevation /data/tmf/rescaled-elevation \
                                                     --slope /data/tmf/rescaled-slopes \
                                                     --access /data/tmf/access \
                                                     --countries-raster /data/tmf/123/countries.tif \
                                                     --output /data/tmf/123/matches
```

Then these rasters get combined into a single raster that is all the potential matches as one:

```shell
$ python3 -m methods.matching.build_m_raster --rasters_directory /data/tmf/123/matches \
                                          --output /data/tmf/123/matches.tif \
                                          -j 20
```

We then convert that raster into a table of pixels plus the data we need:

```shell
$ python3 -m methods.matching.build_m_table --raster /data/tmf/123/matches.tif \
                                            --matching /data/tmf/123/matching-area.geojson \
                                            --start_year 2012 \
                                            --evaluation_year 2021 \
                                            --jrc /data/tmf/jrc/tif/products/tmf_v1/AnnualChange \
                                            --cpc /data/tmf/fcc-cpcs \
                                            --ecoregions /data/tmf/ecoregions \
                                            --elevation /data/tmf/rescaled-elevation \
                                            --slope /data/tmf/rescaled-slopes \
                                            --access /data/tmf/access \
                                            --countries-raster /data/tmf/123/countries.tif \
                                            --output /data/tmf/123/matches.parquet
```

## Find pairs

Finally we can find the 100 sets of matched pairs:

```shell
$ python3 -m methods.matching.find_pairs --k /data/tmf/123/k.parquet \
                                         --m /data/tmf/123/matches.parquet \
                                         --start_year 2012 \
                                         --output /data/tmf/123/pairs \
                                         --seed 42 \
                                         -j 1 \
```

# Calculate additionality

Finally this script calculates additionality:

```shell
$ python -m methods.outputs.calculate_additionality --project /data/tmf/projects/123/boundary.geojson \
                                                    --project_start 2012
                                                    --evaluation_year 2021 \
                                                    --density /data/tmf/123/carbon-density.csv \
                                                    --matches /data/tmf/123/pairs \
                                                    --output /data/tmf/123/additionality.csv
```



```shell
python -m methods.inputs.generate_country_raster --jrc ./inputs/jrc/tif/products/tmf_v1/AnnualChange --matching ./inputs/1201-matching-area.geojson --countries ./inputs/countries.geojson --output ./data/1201-countries.tif
python -m methods.inputs.generate_country_list --leakage ./inputs/1201-leakage.geojson --countries ./inputs/countries.geojson --output ./data/1201-leakage-country-list.json
python -m methods.inputs.generate_matching_area --project ./inputs/1201-leakage.geojson --countrycodes ./inputs/1201-leakage-country-list.json --countries ./inputs/countries.geojson --ecoregions ./inputs/ecoregions.geojson --projects ./inputs/projects --output ./data/1201-leakage-matching-area.geojson
python -m methods.inputs.generate_country_raster --jrc ./inputs/jrc/tif/products/tmf_v1/AnnualChange --matching ./inputs/1201-leakage-matching-area.geojson --countries ./inputs/countries.geojson --output ./data/1201-leakage-countries.tif
python -m methods.inputs.download_srtm_data ./inputs/1201.geojson ./inputs/1201-leakage-matching-area.geojson ./data/srtm_zip ./data/srtm_tif
python -m methods.inputs.generate_slope --input ./inputs/srtm_tif --output ./data/slopes
python -m methods.inputs.rescale_tiles_to_jrc --jrc ./inputs/jrc/tif/products/tmf_v1/AnnualChange --tiles ./inputs/slopes --output ./data/rescaled-slopes
python -m methods.inputs.rescale_tiles_to_jrc --jrc ./inputs/jrc/tif/products/tmf_v1/AnnualChange --tiles ./inputs/srtm_tif --output ./data/rescaled-elevation

```