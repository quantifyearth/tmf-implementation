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

## Make a boundary for generating AGB data

We add a 30km buffer around the project for generating land usage and AGB data:

```shell
$ python3 -m methods.inputs.generate_boundary --project /data/tmf/projects/123/boundary.geojson --output /data/tmf/123/buffer.geojson
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


