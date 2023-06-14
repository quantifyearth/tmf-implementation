Tropical Moist Forest Methodology
---------------------------------

## Building and testing

This project relies on Python 3.10 or newer.

The easiest way to get setup is to use `virtualenv` then install the `requirements.txt`.

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To typecheck the code run `make type` and to run the tests use `make test`.

## RFCs

The RFC documents are submoduled in [tmf-methodology](./tmf-methodology/).