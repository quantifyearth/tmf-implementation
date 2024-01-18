# Running the pipeline

This documents a simple run through of the pipeline for a single project. It assumes you are starting from scratch - most times you don't need to run the full pipeline as you already have the inputs, and just run the matching sections.

Assumptions:

* You have set up your python environment as per the top level README.
* You are executing these commands from the root directory of this repository.
* You have a target directory that can cope with all the many gigabytes of data required.
* You have a POSTGIS database set up for holding the GEDI data.

To make it easier to follow, I will assume you are keeping your data generally in `/data/tmf`

## Downloading resources used by all projects

First we need to download the layers used by all projects, and in some cases convert them into a more useful form.

### JRC

JRC data is generally downloaded first, as all other tiles are at JRC resolution, and so often things that don't need JRC data will still read one JRC tile just to get the correct GeoTIFF projection and pixel scale. To fetch JRC data we do:

```shell
python3 -m methods.inputs.download_jrc_data /data/tmf/jrc/zips /data/tmf/jrc/tif
```

The zips are kept around for archival purposes, due to known difficultly in versioning JRC data, but only the tifs directory is needed from here on.

### ACCESS

Firstly we have access to healthcare data:

```shell
python3 -m methods.inputs.download_accessibility /data/tmf/access/raw.tif
```

Which must then be turned into smaller tiles:

```shell
python3 -m methods.inputs.generate_access_tiles /data/tmf/access/raw.tif /data/tmf/jrc/tif /data/tmf/access
```

### Country boarders

We use OSM data for country board data:

```shell
python3 -m methods.inputs.osm_countries /data/tmf/osm_borders.geojson
```

## Download resources for a specific project

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

###

