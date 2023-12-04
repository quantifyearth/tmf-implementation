Tropical Moist Forest Accreditation Methodology Implementation
--------------------------------------------------------------

This repository contains an implementation of the [PACT Tropical Moist Forest
Accreditation
Methodology](https://www.cambridge.org/engage/coe/article-details/64621025fb40f6b3eea0642f).

From version 2.0.0 of the methodolgy, the git tags are synchronised with the
corresponding versions of the methodology document (not necessarily the version
number on the open-engage platform).

## Development

This project relies on Python 3.10 or newer.

The easiest way to get setup is to use `virtualenv` then install the
`requirements.txt`.

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To typecheck the code run `make type` and to run the tests use `make test`.

### System Requirements

Some stages of the pipeline require significant parallelisation to complete
quickly which tends to also require plenty of memory too.

### Structure

The code is broken into three main sections contained in the
[methods](./methods) directory.

 1. Inputs: the scripts in the [inputs](./methods/inputs/) directory are used to
    download the necessary input data to run the methodology. For example, the
    [land use class data from the JRC TMF
    dataset](./methods/inputs/download_jrc_data.py). Some of these scripts also
    do some light processing too, for example, [to generate slopes from
    elevations maps](./methods/inputs/generate_slope.py).
 2. Matching: the scripts in the [matching](./methods/matching/) directory are
    for performing the _counterfactual pixel matching_ algorithms described in
    the   methodology document. Broadly speaking, this first find [potential
    candidates (referred to as the set `M` of
    pixels)](./methods/matching/find_potential_matches.py) and then finds actual
    pairs from pixels in the treatment area to pixels in the control area.
 3. Outputs: the scripts in the [outputs](./methods/outputs/) directory are used
    to generate the outputs from the methodology like the equivalent permanence
    and the additionality per year.

## Bugs

Should you find any bugs or issues with the code then please do open an issue on
the [github issue
tracker](https://github.com/quantifyearth/tmf-implementation/issues/new).